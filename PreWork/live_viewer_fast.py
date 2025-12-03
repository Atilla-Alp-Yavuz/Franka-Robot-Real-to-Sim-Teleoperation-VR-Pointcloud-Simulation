import time
import cv2
import numpy as np
import mujoco
from mujoco import viewer
from pathlib import Path
import threading
import queue

# ----------------------------------
# CONFIG
# ----------------------------------
W, H = 320, 240        # lower resolution = much faster
CAMERAS = ["top", "front", "left", "right"]
SHOW_DEPTH = False     # toggle depth / rgb
FPS = 30               # display FPS target

# ----------------------------------
# LOAD MODEL
# ----------------------------------
xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

model.vis.global_.offwidth = W
model.vis.global_.offheight = H

# Queue for passing frames to the display thread
frame_queue = queue.Queue(maxsize=1)


# ----------------------------------
# SIMPLE DEPTH VIS (fast)
# ----------------------------------
def depth_to_vis(depth):
    depth = np.nan_to_num(depth, nan=5.0)
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    d = (255 * d).astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)


# ----------------------------------
# DISPLAY THREAD (OpenCV)
# ----------------------------------
def display_loop():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        cv2.imshow("FAST Multi-Camera Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Start display thread
display_thread = threading.Thread(target=display_loop, daemon=True)
display_thread.start()


# ----------------------------------
# MAIN SIMULATION + RENDERING LOOP
# ----------------------------------
with viewer.launch_passive(model, data) as v:
    with mujoco.Renderer(model, width=W, height=H) as renderer:

        t0 = time.time()
        last_display_time = 0

        while v.is_running():

            # -------------------------------
            # ROBOT DUMMY MOTION
            # -------------------------------
            t = time.time() - t0
            data.qpos[0] = initial_qpos[0] + 0.3 * np.sin(0.5 * t)
            data.qpos[1] = initial_qpos[1] + 0.2 * np.sin(0.8 * t)
            data.qpos[3] = initial_qpos[3] + 0.15 * np.sin(1.2 * t)

            mujoco.mj_step(model, data)

            # Update MuJoCo viewer separately
            v.sync()

            # Limit display FPS (don't update every frame)
            now = time.time()
            if now - last_display_time < 1.0 / FPS:
                continue
            last_display_time = now

            # -------------------------------
            # RENDER CAMERAS (lightweight)
            # -------------------------------
            frames = []
            for cam in CAMERAS:

                renderer.update_scene(data, camera=cam)

                if SHOW_DEPTH:
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=cam)
                    depth = renderer.render()
                    renderer.disable_depth_rendering()
                    frame = depth_to_vis(depth)
                else:
                    rgb = renderer.render()
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                frames.append(frame)

            # -------------------------------
            # BUILD 2×2 TILE
            # -------------------------------
            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            tiled = np.vstack((top_row, bottom_row))

            # -------------------------------
            # PUSH TO DISPLAY THREAD
            # -------------------------------
            if not frame_queue.full():
                frame_queue.put(tiled)


# Cleanup
frame_queue.put(None)
display_thread.join()
