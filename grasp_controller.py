import mujoco
import mujoco.viewer
import numpy as np
import cv2

#  LOAD MODEL
model = mujoco.MjModel.from_xml_path("mjxpandamerged.xml")
data  = mujoco.MjData(model)

#  ACTUATOR INDEX MAP
#  ctrl[0–6]   = Panda joints 1–7
#  ctrl[7–10]  = Index  (ffa0 ffa1 ffa2 ffa3)
#  ctrl[11–14] = Middle (mfa0 mfa1 mfa2 mfa3)
#  ctrl[15–18] = Ring   (rfa0 rfa1 rfa2 rfa3)
#  ctrl[19–22] = Thumb  (tha0 tha1 tha2 tha3)
PANDA  = slice(0,  7)

# String keys used for all per-finger dicts
FINGER_SLICES = {
    "ff": slice(7,  11),
    "mf": slice(11, 15),
    "rf": slice(15, 19),
    "th": slice(19, 23),
}

#  FINGER CONTACT DETECTION
#  Uses MuJoCo contact array, checks if each fingertip geom is in contact with the brick
FINGER_TIPS = {
    "ff": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ff_tip"),
    "mf": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mf_tip"),
    "rf": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rf_tip"),
    "th": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "th_tip"),
}
BRICK_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brick")

def get_contacts(data):
    contacts = {f: False for f in FINGER_TIPS}
    for i in range(data.ncon):
        con = data.contact[i]
        # Each contact stores two geom ids; look up which bodies they belong to
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        bodies = {b1, b2}
        if BRICK_BODY in bodies:
            for name, tip_id in FINGER_TIPS.items():
                if tip_id in bodies:
                    contacts[name] = True
    return contacts

#  GELSIGHT-LIKE TACTILE DEPTH MAP
#
#  Instead of a raw camera render (eye-in-hand), there is a 2D depth camera from MuJoCo's contact data directly
#  Each contact point is projected into the fingertip's local frame and splattered onto a 2D grid with a Gaussian kernel
#  This produces an image where: bright pixels = depth / firm contact ; dark pixels = no contact
IMG_SIZE    = 64      # pixels per axis
PAD_RADIUS  = 0.012   # physical size of sensing area (meters)
MAX_DEPTH   = 0.005   # depth range mapped to full brightness

# Precompute Gaussian kernel for contact splatting
_ks = 5               # kernel half-size
_kernel = np.zeros((2*_ks+1, 2*_ks+1), dtype=np.float32)
for _dy in range(-_ks, _ks+1):
    for _dx in range(-_ks, _ks+1):
        _kernel[_dy+_ks, _dx+_ks] = np.exp(-(_dx**2+_dy**2)/8.0)

def get_depth_image(data, finger_name):
    tip_id  = FINGER_TIPS[finger_name]
    tip_pos = data.xpos[tip_id]
    tip_rot = data.xmat[tip_id].reshape(3, 3)

    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for i in range(data.ncon):
        con = data.contact[i]
        b1  = model.geom_bodyid[con.geom1]
        b2  = model.geom_bodyid[con.geom2]
        if tip_id not in (b1, b2):
            continue

        # Project contact position into fingertip local frame
        local = tip_rot.T @ (con.pos - tip_pos)

        # local[0], local[1] = lateral position on pad
        # local[2]           = depth into pad
        px = int((local[0] / PAD_RADIUS + 0.5) * IMG_SIZE)
        py = int((local[1] / PAD_RADIUS + 0.5) * IMG_SIZE)

        if not (0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE):
            continue

        # Intensity encodes penetration depth
        intensity = float(np.clip(1.0 - abs(local[2]) / MAX_DEPTH, 0.0, 1.0))

        # Splat Gaussian blob (simulates gel spreading around contact)
        x0 = max(0, px - _ks);  x1 = min(IMG_SIZE, px + _ks + 1)
        y0 = max(0, py - _ks);  y1 = min(IMG_SIZE, py + _ks + 1)
        kx0 = x0 - (px - _ks); kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (py - _ks); ky1 = ky0 + (y1 - y0)
        img[y0:y1, x0:x1] = np.maximum(
            img[y0:y1, x0:x1],
            _kernel[ky0:ky1, kx0:kx1] * intensity
        )

    return img

def colorize_depth(img, contact):
    grey   = (img * 255).astype(np.uint8)
    color  = cv2.applyColorMap(grey, cv2.COLORMAP_JET)

    # Draw a circle to indicate the physical pad boundary
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    cv2.circle(color, (cx, cy), IMG_SIZE // 2 - 2,
               (0, 200, 0) if contact else (80, 80, 80), 2)

    return color

FINGERS = ["ff", "mf", "rf", "th"]

def get_tactile_row(data, contacts):
    images = []
    for finger in FINGERS:
        depth = get_depth_image(data, finger)
        img   = colorize_depth(depth, contacts[finger])

        # Scale up for visibility and add label
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        border_color = (0, 220, 0) if contacts[finger] else (100, 100, 100)
        cv2.rectangle(img, (0, 0), (127, 127), border_color, 3)
        cv2.putText(img, finger.upper(), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
        images.append(img)

    row = np.concatenate(images, axis=1)
    return cv2.resize(row, (row.shape[1] * 2, row.shape[0] * 2),
                      interpolation=cv2.INTER_NEAREST)

#  POSE DEFINITIONS

# Panda: home pose (arm extended, hand above table)
PANDA_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])

# Panda: pre-grasp pose = hand lowered in front, oriented for top-down pinch
# These values position the Allegro palm roughly above a brick at ~(0.5, 0, 0.05)
PANDA_PREGRASP = np.array([0.0, 1.0, 0.0, -3.0, 0.0, 3.0, 0.785])

# Allegro: fully open = spread fingers wide
HAND_OPEN = {
    "ff": np.array([ 0.0,  0.15, 0.15, 0.15]),   # base spread, slight curl
    "mf": np.array([ 0.0,  0.15, 0.15, 0.15]),
    "rf": np.array([ 0.0,  0.15, 0.15, 0.15]),
    "th": np.array([ 0.8,  0.4,  0.3,  0.2 ]),   # thumb pre-positioned
}

# Allegro: pinch target = fingers curl inward toward thumb
# For a fingertip pinch the proximal/medial/distal joints close
PINCH_TARGET = {
    "ff": np.array([ 0.15, 0.8,  0.8,  0.8 ]),
    "mf": np.array([ 0.0,  0.8,  0.8,  0.8 ]),
    "rf": np.array([-0.15, 0.8,  0.8,  0.8 ]),
    "th": np.array([ 0.9,  0.8,  0.7,  0.6 ]),
}

#  INTERPOLATION HELPER
def lerp(a, b, t):
    t = np.clip(t, 0.0, 1.0)
    return a + (b - a) * t

#  CONTROLLER STATE MACHINE
# Phases:
#   0 = move Panda arm to pre-grasp pose
#   1 = open hand
#   2 = close fingers until all four tips contact brick
#   3 = grasp held (fingers freeze at contact pose)

class GraspController:
    def __init__(self):
        self.phase        = 0
        self.phase_timer  = 0
        self.frozen       = {f: False for f in FINGER_SLICES}
        self.frozen_ctrl  = {}   # keyed by finger name string

    def step(self, data):
        contacts = get_contacts(data)
        self.phase_timer += 1

        # Phase 0: move arm to pre-grasp
        if self.phase == 0:
            t = self.phase_timer / 300.0
            data.ctrl[PANDA] = lerp(PANDA_HOME, PANDA_PREGRASP, t)
            for fname, slc in FINGER_SLICES.items():
                data.ctrl[slc] = HAND_OPEN[fname]
            if self.phase_timer >= 350:
                print("[Phase 0 complete] Arm at pre-grasp pose")
                self.phase = 1
                self.phase_timer = 0

        # Phase 1: open hand fully 
        elif self.phase == 1:
            t = self.phase_timer / 100.0
            for fname, slc in FINGER_SLICES.items():
                data.ctrl[slc] = lerp(data.ctrl[slc], HAND_OPEN[fname], t)
            if self.phase_timer >= 120:
                print("[Phase 1 complete] Hand open = beginning pinch close")
                self.phase = 2
                self.phase_timer = 0

        # Phase 2: close fingers until contact 
        elif self.phase == 2:
            t = self.phase_timer / 600.0

            for fname, slc in FINGER_SLICES.items():
                if not self.frozen[fname]:
                    if contacts[fname]:
                        self.frozen[fname]         = True
                        self.frozen_ctrl[fname]    = data.ctrl[slc].copy()
                        print(f"  [{fname.upper()} contact detected] = finger frozen")
                    else:
                        data.ctrl[slc] = lerp(HAND_OPEN[fname],
                                               PINCH_TARGET[fname], t)
                else:
                    data.ctrl[slc] = self.frozen_ctrl[fname]

            data.ctrl[PANDA] = PANDA_PREGRASP

            all_contact = all(self.frozen.values())
            if all_contact:
                print("[Phase 2 complete] All fingers in contact = grasp held")
                self.phase = 3
                self.phase_timer = 0
            elif self.phase_timer >= 700:
                print("[Phase 2 timeout] Not all fingers contacted = holding current pose")
                self.phase = 3
                self.phase_timer = 0

        # Phase 3: hold grasp 
        elif self.phase == 3:
            data.ctrl[PANDA] = PANDA_PREGRASP
            for fname, slc in FINGER_SLICES.items():
                if fname in self.frozen_ctrl:
                    data.ctrl[slc] = self.frozen_ctrl[fname]

        return contacts

#  MAIN LOOP
mujoco.mj_resetData(model, data)

# Set arm to home pose immediately so it doesn't flop
data.ctrl[PANDA] = PANDA_HOME
for fname, slc in FINGER_SLICES.items():
    data.ctrl[slc] = HAND_OPEN[fname]

controller = GraspController()

print("=" * 55)
print("  Grasp Controller - Fingertip Pinch with GelSight")
print("=" * 55)
print("  Phase 0: Panda arm moves to pre-grasp pose")
print("  Phase 1: Hand opens")
print("  Phase 2: Fingers close until tactile contact")
print("  Phase 3: Grasp held at contact pose")
print("-" * 55)
print("  ESC in the tactile window to quit")
print("=" * 55)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        contacts = controller.step(data)

        viewer.sync()

        # Update tactile window
        tactile_row = get_tactile_row(data, contacts)

        # Status bar
        phase_labels = {
            0: "Phase 0: Arm moving to pre-grasp",
            1: "Phase 1: Opening hand",
            2: "Phase 2: Closing — waiting for contact",
            3: "Phase 3: Grasp held",
        }
        label = phase_labels[controller.phase]
        contact_str = "  ".join(
            [f"{f.upper()}:{'ON ' if contacts[f] else 'off'}" for f in FINGERS]
        )
        cv2.putText(tactile_row, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(tactile_row, contact_str, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("GelSight  [FF | MF | RF | TH]", tactile_row)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()