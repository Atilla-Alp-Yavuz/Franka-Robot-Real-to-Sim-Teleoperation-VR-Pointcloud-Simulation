import time
import cv2
import numpy as np
import mujoco
from mujoco import viewer
from pathlib import Path

# ----------------------------------
# CONFIG
# ----------------------------------
W, H = 640, 480
CAMERAS = ["top", "front", "left", "right"]
SHOW_DEPTH = True       # Set True to view depth instead of RGB

# ----------------------------------
# LOAD MODEL
# ----------------------------------
xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Initial pose
initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

model.vis.global_.offwidth = W
model.vis.global_.offheight = H

# ----------------------------------
# DEPTH VISUALIZATION
# ----------------------------------
def depth_to_vis(depth):
    depth = np.nan_to_num(depth, nan=5.0)

    inv = 1.0 / (depth + 1e-6)
    inv -= inv.min()
    inv /= inv.max() + 1e-6
    inv = (inv * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    inv_eq = clahe.apply(inv)

    return inv_eq


# ----------------------------------
# REAL-TIME LOOP
# ----------------------------------
with viewer.launch_passive(model, data) as v:
    with mujoco.Renderer(model, H, W) as renderer:

        t0 = time.time()

        while v.is_running():

            # ----------------------------------
            # ROBOT MOTION
            # ----------------------------------
            t = time.time() - t0
            data.qpos[0] = initial_qpos[0] + 0.3 * np.sin(0.5 * t)
            data.qpos[1] = initial_qpos[1] + 0.2 * np.sin(0.8 * t)
            data.qpos[3] = initial_qpos[3] + 0.15 * np.sin(1.2 * t)

            mujoco.mj_step(model, data)

            # ----------------------------------
            # RENDER ALL CAMERAS
            # ----------------------------------
            views = []

            for cam in CAMERAS:

                renderer.update_scene(data, camera=cam)

                if SHOW_DEPTH:
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=cam)
                    depth = renderer.render()
                    renderer.disable_depth_rendering()

                    frame = depth_to_vis(depth)

                    # Convert 1-channel → 3-channel for tiling
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                else:
                    rgb = renderer.render()
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                views.append(frame)

            # ----------------------------------
            # TILE 2×2 VIEW (top, front, left, right)
            # ----------------------------------
            top_left     = views[0]
            top_right    = views[1]
            bottom_left  = views[2]
            bottom_right = views[3]

            top_row    = np.hstack((top_left, top_right))
            bottom_row = np.hstack((bottom_left, bottom_right))

            tiled = np.vstack((top_row, bottom_row))

            # ----------------------------------
            # DISPLAY LIVE WINDOW
            # ----------------------------------
            cv2.imshow("MuJoCo Multi-Camera Viewer", tiled)

            # Allow quitting with Q key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            v.sync()

cv2.destroyAllWindows()
