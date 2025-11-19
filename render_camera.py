import mujoco
from mujoco import viewer
from pathlib import Path
import numpy as np
import cv2

xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

model.vis.global_.offwidth = 640
model.vis.global_.offheight = 480

width, height = 640, 480
renderer = mujoco.Renderer(model, width=640, height=480)

initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

renderer.update_scene(data, camera="top")

rgb = renderer.render()
depth = renderer.render()

cv2.imwrite("top_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
cv2.imwrite("top_depth.png", (depth_norm * 255).astype(np.uint8))

print("Saved top_rgb.png and top_depth.png")
