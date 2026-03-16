import argparse
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from brick_lift_env import BrickLiftEnv, FINGERS, IMG_SIZE, PAD_RADIUS, MAX_DEPTH

#  TACTILE DISPLAY  (same as grasp_controller)
_ks = 3
import numpy as _np
_kernel = _np.zeros((2*_ks+1, 2*_ks+1), dtype=_np.float32)
for _dy in range(-_ks, _ks+1):
    for _dx in range(-_ks, _ks+1):
        _kernel[_dy+_ks, _dx+_ks] = _np.exp(-(_dx**2+_dy**2)/4.0)


def colorize_depth(img, contact, size=128):
    grey  = (img * 255).astype(np.uint8)
    color = cv2.applyColorMap(
        cv2.resize(grey, (size, size), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_JET
    )
    cx = cy = size // 2
    cv2.circle(color, (cx, cy), cx - 2,
               (0, 200, 0) if contact else (80, 80, 80), 2)
    return color


def get_tactile_display(env):
    images = []
    for finger in FINGERS:
        depth   = env._get_depth_image(finger)
        contact = any(
            env.model.geom_bodyid[env.data.contact[i].geom1] ==
            env.finger_tips[finger] or
            env.model.geom_bodyid[env.data.contact[i].geom2] ==
            env.finger_tips[finger]
            for i in range(env.data.ncon)
        )
        img = colorize_depth(depth, contact)
        cv2.putText(img, finger.upper(), (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 0) if contact else (150, 150, 150), 1)
        images.append(img)

    row = np.concatenate(images, axis=1)
    return cv2.resize(row, (row.shape[1] * 2, row.shape[0] * 2),
                      interpolation=cv2.INTER_NEAREST)


def evaluate(model_path="best_model/best_model", n_episodes=10):
    print(f"  Loading model from: {model_path}")

    # Build env
    raw_env = BrickLiftEnv(xml_path="mjxpandamerged.xml", render_mode="human")
    vec_env = DummyVecEnv([lambda: raw_env])

    # Load normalisation stats if available
    import os
    norm_path = "checkpoints/vec_normalize.pkl"
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False   # don't update stats during eval
        vec_env.norm_reward = False
        print("  Loaded observation normalisation stats.")

    model = PPO.load(model_path, env=vec_env)
    print("  Model loaded. Running evaluation...\n")

    successes = 0

    for ep in range(n_episodes):
        obs, _   = raw_env.reset()
        vec_obs  = vec_env.reset()
        done     = False
        ep_reward = 0.0
        step      = 0

        with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
            while not done and viewer.is_running():
                action, _ = model.predict(vec_obs, deterministic=True)
                vec_obs, reward, done_arr, info = vec_env.step(action)
                done = bool(done_arr[0])
                ep_reward += float(reward[0])
                step += 1

                viewer.sync()

                # Tactile window
                tactile = get_tactile_display(raw_env)
                brick_h = raw_env.data.xpos[raw_env.brick_body][2]
                cv2.putText(tactile,
                            f"Episode {ep+1}  step {step:4d}  "
                            f"height={brick_h:.3f}m",
                            (10, tactile.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0), 1)
                cv2.imshow("GelSight Tactile  [FF | MF | RF | TH]", tactile)
                if cv2.waitKey(1) == 27:
                    done = True

        if ep_reward >= 100:
            successes += 1
            print(f"  Episode {ep+1:2d}: ✓ SUCCESS  reward={ep_reward:.1f}")
        else:
            print(f"  Episode {ep+1:2d}: ✗ failed   reward={ep_reward:.1f}")

    print(f"\n  Success rate: {successes}/{n_episodes} "
          f"({successes/n_episodes:.0%})")
    cv2.destroyAllWindows()
    raw_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model/best_model",
                        help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    evaluate(model_path=args.model, n_episodes=args.episodes)