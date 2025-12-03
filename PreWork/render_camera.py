import mujoco
from pathlib import Path
import numpy as np
import cv2

# ------------ Config ------------
W, H = 640, 480
CAMERAS = ["top", "front", "left", "right"]

# ------------ Load model ------------
xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Initial pose
initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

# Increase offscreen buffer resolution
model.vis.global_.offwidth = W
model.vis.global_.offheight = H

def enhance_depth_very_detailed(depth: np.ndarray) -> np.ndarray:
    """
    Turn a MuJoCo depth map into a very detailed visualization:
    - inverse depth
    - adaptive histogram equalization (CLAHE)
    - light sharpening
    Returns uint8 image (H x W).
    """
    # 1) Remove NaNs and absurd values
    depth = np.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=0.0)

    # 2) Inverse depth to amplify small z-changes
    inv = 1.0 / (depth + 1e-6)
    inv -= inv.min()
    if inv.max() > 0:
        inv /= inv.max()

    inv_uint8 = (inv * 255).astype(np.uint8)

    # 3) Local contrast enhancement (CLAHE) - key for "more detail"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    inv_eq = clahe.apply(inv_uint8)

    # 4) Gentle sharpening (unsharp mask)
    blur = cv2.GaussianBlur(inv_eq, (5, 5), 1.0)
    sharpened = cv2.addWeighted(inv_eq, 1.5, blur, -0.5, 0)

    return sharpened

# ------------ Helper: make highly detailed depth map ------------
def enhance_depth(depth):
    """Convert MuJoCo depth → high-detail visualization"""

    # --- (1) Clean NaNs ---
    depth = np.nan_to_num(depth, nan=5.0)

    # --- (2) Inverse depth (1/z) for detail enhancement ---
    inv = 1.0 / (depth + 1e-6)
    inv /= inv.max()  # normalize

    # --- (3) Histogram equalization (massive detail increase) ---
    inv_uint8 = (inv * 255).astype(np.uint8)
    inv_eq = cv2.equalizeHist(inv_uint8)  # spreads pixel intensities

    # --- (4) Slight blur to reduce banding ---
    inv_eq = cv2.GaussianBlur(inv_eq, (5, 5), sigmaX=1.0)

    return inv_eq  # uint8 image


with mujoco.Renderer(model, width=W, height=H) as renderer:
    for cam in CAMERAS:
        print(f"[Rendering]: {cam}")

        # RGB
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        cv2.imwrite(f"{cam}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render()
        renderer.disable_depth_rendering()

        ##CHANGE THIS FUNCTION ADJUST WHEN MORE DEPTH NEEDED
        #IMPORTANTTT
        depth_detail = enhance_depth_very_detailed(depth)
        cv2.imwrite(f"{cam}_depth.png", depth_detail)

print("DONE — saved very detailed depth maps.")
