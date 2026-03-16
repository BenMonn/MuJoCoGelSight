import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

#  CONSTANTS  (must match grasp_controller.py)
PANDA        = slice(0, 7)
FINGER_SLICES = {
    "ff": slice(7,  11),
    "mf": slice(11, 15),
    "rf": slice(15, 19),
    "th": slice(19, 23),
}
FINGERS = ["ff", "mf", "rf", "th"]

# Grasp controller pose — RL starts from here
PANDA_PREGRASP = np.array([0.0, 0.6, 0.0, -2.2, 0.0, 2.1, 0.785])

HAND_OPEN = {
    "ff": np.array([ 0.0,  0.15, 0.15, 0.15]),
    "mf": np.array([ 0.0,  0.15, 0.15, 0.15]),
    "rf": np.array([ 0.0,  0.15, 0.15, 0.15]),
    "th": np.array([ 0.8,  0.4,  0.3,  0.2 ]),
}

# RL episode parameters
LIFT_HEIGHT       = 0.12   # metres above floor — counts as "lifted"
HOLD_SECONDS      = 2.0    # seconds brick must stay at height
SIM_TIMESTEP      = 0.002  # must match XML timestep (default MuJoCo)
STEPS_PER_ACTION  = 5      # physics steps per RL action (control freq)
MAX_EPISODE_STEPS = 500    # max steps before episode ends

# Tactile image size (must match grasp_controller.py)
IMG_SIZE   = 16    # kept small for RL obs vector — 16x16 per finger
PAD_RADIUS = 0.012
MAX_DEPTH  = 0.005

# Gaussian kernel for tactile splatting
_ks = 3
_kernel = np.zeros((2*_ks+1, 2*_ks+1), dtype=np.float32)
for _dy in range(-_ks, _ks+1):
    for _dx in range(-_ks, _ks+1):
        _kernel[_dy+_ks, _dx+_ks] = np.exp(-(_dx**2+_dy**2)/4.0)


class BrickLiftEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path="mjxpandamerged.xml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_TIMESTEP

        # Body / joint IDs
        self.brick_body  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "brick")
        self.finger_tips = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip")
            for f in FINGERS
        }

        # Brick qpos address (freejoint = 7 values: pos + quat)
        # Find by body since the freejoint has no name in the XML
        self.brick_qpos_adr = None
        for jnt_id in range(self.model.njnt):
            if self.model.jnt_bodyid[jnt_id] == self.brick_body:
                self.brick_qpos_adr = self.model.jnt_qposadr[jnt_id]
                break
        if self.brick_qpos_adr is None:
            raise RuntimeError("Could not find a joint on the brick body. Ensure brick has <freejoint/> in the XML.")

        # Hold counter
        self.hold_steps    = 0
        self.hold_required = int(HOLD_SECONDS / (SIM_TIMESTEP * STEPS_PER_ACTION))

        # Observation space 
        obs_dim = (
            23 +              # joint positions
            23 +              # joint velocities
            3  +              # brick xyz
            4  +              # brick quaternion
            4 * IMG_SIZE**2   # tactile depth maps
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space
        # Residual deltas on the 16 Allegro joints, clamped to +/-0.1 rad
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(16,), dtype=np.float32)

        # Store the grasp pose the hand was in when the episode starts
        self._grasp_ctrl = None

        # Renderer for human mode
        self._viewer = None

    #  RESET
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Arm at pre-grasp
        self.data.ctrl[PANDA] = PANDA_PREGRASP
        self.data.qpos[0:7]   = PANDA_PREGRASP

        # Hand open
        for fname, slc in FINGER_SLICES.items():
            self.data.ctrl[slc] = HAND_OPEN[fname]

        # Place brick at a reachable position with small random offset so the policy learns to generalise slightly
        rng = np.random.default_rng(seed)
        brick_xy_noise = rng.uniform(-0.01, 0.01, size=2)
        self.data.qpos[self.brick_qpos_adr:self.brick_qpos_adr+3] = \
            np.array([0.5, 0.0, 0.025]) + np.append(brick_xy_noise, 0.0)
        # Identity quaternion (upright brick)
        self.data.qpos[self.brick_qpos_adr+3:self.brick_qpos_adr+7] = \
            np.array([1.0, 0.0, 0.0, 0.0])

        # Step forward to let the arm settle into pre-grasp pose
        self.data.ctrl[PANDA] = PANDA_PREGRASP
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Close fingers to initial grasp (same as grasp controller phase 2)
        PINCH_START = {
            "ff": np.array([ 0.15, 0.6, 0.6, 0.6]),
            "mf": np.array([ 0.0,  0.6, 0.6, 0.6]),
            "rf": np.array([-0.15, 0.6, 0.6, 0.6]),
            "th": np.array([ 0.9,  0.7, 0.6, 0.5]),
        }
        for fname, slc in FINGER_SLICES.items():
            self.data.ctrl[slc] = PINCH_START[fname]
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Save the grasp ctrl as the RL baseline
        self._grasp_ctrl = self.data.ctrl[7:23].copy()

        self.hold_steps = 0
        self._step_count = 0

        return self._get_obs(), {}

    #  STEP
    def step(self, action):
        # Apply residual action on top of current hand ctrl
        new_hand_ctrl = np.clip(
            self.data.ctrl[7:23] + action,
            self.model.actuator_ctrlrange[7:23, 0],
            self.model.actuator_ctrlrange[7:23, 1]
        )
        self.data.ctrl[7:23] = new_hand_ctrl
        # Arm stays fixed
        self.data.ctrl[PANDA] = PANDA_PREGRASP

        # Step physics
        for _ in range(STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs          = self._get_obs()
        brick_height = self.data.xpos[self.brick_body][2]
        reward, terminated = self._compute_reward(brick_height, action)
        truncated    = self._step_count >= MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, {}

    #  OBSERVATION
    def _get_obs(self):
        # Joint positions and velocities (all 23 DOF)
        qpos = self.data.qpos[0:23].astype(np.float32)
        qvel = self.data.qvel[0:23].astype(np.float32)

        # Brick state
        brick_pos  = self.data.xpos[self.brick_body].astype(np.float32)
        brick_quat = self.data.xquat[self.brick_body].astype(np.float32)

        # Tactile depth maps — one per finger, flattened
        tactile = np.concatenate([
            self._get_depth_image(f).flatten() for f in FINGERS
        ]).astype(np.float32)

        return np.concatenate([qpos, qvel, brick_pos, brick_quat, tactile])

    def _get_depth_image(self, finger_name):
        tip_id  = self.finger_tips[finger_name]
        tip_pos = self.data.xpos[tip_id]
        tip_rot = self.data.xmat[tip_id].reshape(3, 3)
        img     = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1  = self.model.geom_bodyid[con.geom1]
            b2  = self.model.geom_bodyid[con.geom2]
            if tip_id not in (b1, b2):
                continue
            local = tip_rot.T @ (con.pos - tip_pos)
            px = int((local[0] / PAD_RADIUS + 0.5) * IMG_SIZE)
            py = int((local[1] / PAD_RADIUS + 0.5) * IMG_SIZE)
            if not (0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE):
                continue
            intensity = float(np.clip(1.0 - abs(local[2]) / MAX_DEPTH, 0, 1))
            x0 = max(0, px-_ks); x1 = min(IMG_SIZE, px+_ks+1)
            y0 = max(0, py-_ks); y1 = min(IMG_SIZE, py+_ks+1)
            kx0 = x0-(px-_ks); kx1 = kx0+(x1-x0)
            ky0 = y0-(py-_ks); ky1 = ky0+(y1-y0)
            img[y0:y1, x0:x1] = np.maximum(
                img[y0:y1, x0:x1],
                _kernel[ky0:ky1, kx0:kx1] * intensity
            )
        return img

    #  REWARD
    def _compute_reward(self, brick_height, action):
        reward     = 0.0
        terminated = False

        # Continuous height reward — always pushing upward
        reward += brick_height * 10.0

        # Bonus for being above lift threshold
        if brick_height > LIFT_HEIGHT:
            reward += 10.0
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        # Success = held at height for required duration
        if self.hold_steps >= self.hold_required:
            reward    += 100.0
            terminated = True
            print(f"  ✓ SUCCESS — brick held at {brick_height:.3f}m "
                  f"for {HOLD_SECONDS}s")

        # Time penalty
        reward -= 1.0

        # Smooth control penalty
        reward -= 0.01 * float(np.sum(action**2))

        # Failure = brick fell off the table
        if brick_height < -0.05:
            reward    -= 50.0
            terminated = True

        return reward, terminated

    #  RENDER
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None