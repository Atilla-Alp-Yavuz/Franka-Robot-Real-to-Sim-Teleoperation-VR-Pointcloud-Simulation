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
initial_qpos = [0.0, -0.5, 0.0, -1.2, 0.0, 1.6, 0.0]
data.qpos[:len(initial_qpos)] = initial_qpos
mujoco.mj_forward(model, data)

# ------------ Inspect camera ------------
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
print("Camera ID:", cam_id)
print("Original cam_pos:", model.cam_pos[cam_id])
print("Original cam_quat:", model.cam_quat[cam_id])

# ------------ Force camera pose ------------
# Put camera above the robot, looking down
model.cam_pos[cam_id] = np.array([0.0, 0.0, 1.2])      # x, y, z
model.cam_quat[cam_id] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

# Safer near/far for depth
model.vis.map.znear = 0.01
model.vis.map.zfar = 5.0

# Ensure offscreen buffer big enough
model.vis.global_.offwidth = W
model.vis.global_.offheight = H

# Recompute after changing camera settings
mujoco.mj_forward(model, data)

print("Adjusted cam_pos:", model.cam_pos[cam_id])
print("Adjusted cam_quat:", model.cam_quat[cam_id])

# ------------ Render RGB + Depth ------------
with mujoco.Renderer(model, width=W, height=H) as renderer:
    # --- RGB ---
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render()  # H x W x 3, uint8
    print("RGB min/max:", rgb.min(), rgb.max())

    # --- Depth ---
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME)
    depth = renderer.render()  # H x W, float32, meters
    renderer.disable_depth_rendering()
    print("Depth min/max:", float(depth.min()), float(depth.max()))

# ------------ Save images ------------
# Save RGB image
cv2.imwrite("top_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Save depth visualization if it’s not constant
if np.isclose(depth.max(), depth.min()):
    print("Warning: depth is constant (camera may see nothing meaningful), not saving depth image.")
else:
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite("top_depth.png", depth_img)

print("Done. Check top_rgb.png and (if created) top_depth.png")