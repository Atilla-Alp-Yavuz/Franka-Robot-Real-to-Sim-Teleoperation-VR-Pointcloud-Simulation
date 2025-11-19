import mujoco
from pathlib import Path
import numpy as np
import cv2

# ------------ Config ------------
W, H = 640, 480
CAMERA_NAME = "top"

# ------------ Load model & data ------------
xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# (Optional) nicer initial pose
initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

# Ensure offscreen buffer big enough
model.vis.global_.offwidth = W
model.vis.global_.offheight = H

# ------------ Render RGB + Depth ------------
with mujoco.Renderer(model, width=W, height=H) as renderer:
    # --- RGB ---
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render()                          # H x W x 3, uint8

    # --- Depth ---
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME)
    depth = renderer.render()                        # H x W, float32, meters
    renderer.disable_depth_rendering()

# Save RGB image
cv2.imwrite("top_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Save depth visualization
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
cv2.imwrite("top_depth.png", (depth_norm * 255).astype(np.uint8))

print("Saved top_rgb.png and top_depth.png")
