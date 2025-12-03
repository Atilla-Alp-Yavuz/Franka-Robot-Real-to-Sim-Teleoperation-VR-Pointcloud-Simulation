import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from mujoco import viewer

# -----------------------
# CONFIG
# -----------------------
W, H = 320, 240
CAMERAS = ["top", "front", "left", "right"]
SHOW_DEPTH = False        # True → show depth instead of RGB
FPS = 30

LAB_XMLS = {
    0: "lab0.xml",
    1: "lab1_T_stack.xml",
    2: "lab2_boxes_cups.xml",
    3: "lab3_stick_maze.xml",
}


def load_lab(lab_id: int, W: int, H: int):
    xml_path = Path(LAB_XMLS[lab_id])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    return model, data


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Simple depth → BGR visualization."""
    depth = np.nan_to_num(depth, nan=5.0)
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    d = (255 * d).astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, data = load_lab(lab_id, W, H)
    print(f"Loaded lab {lab_id}: {LAB_XMLS[lab_id]}")

    # Create OpenCV window in the MAIN thread
    cv2.namedWindow("LAB Multi-Camera Viewer", cv2.WINDOW_NORMAL)

    with viewer.launch_passive(model, data) as v:
        with mujoco.Renderer(model, width=W, height=H) as renderer:

            t0 = time.time()
            last_display_time = 0.0

            while v.is_running():
                # ---- YOUR CONTROL / TELEOP GOES HERE ----
                # For now: just keep robot static or add tiny motion if you want
                mujoco.mj_step(model, data)

                # Keep MuJoCo 3D viewer in sync
                v.sync()

                # Limit display FPS
                now = time.time()
                if now - last_display_time < 1.0 / FPS:
                    continue
                last_display_time = now

                # ---- Render cameras ----
                frames = []
                for cam in CAMERAS:
                    if SHOW_DEPTH:
                        renderer.enable_depth_rendering()
                        renderer.update_scene(data, camera=cam)
                        depth = renderer.render()
                        renderer.disable_depth_rendering()
                        frame = depth_to_vis(depth)
                    else:
                        renderer.update_scene(data, camera=cam)
                        rgb = renderer.render()
                        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    frames.append(frame)

                # ---- Tile 2×2 ----
                top_row = np.hstack((frames[0], frames[1]))
                bottom_row = np.hstack((frames[2], frames[3]))
                tiled = np.vstack((top_row, bottom_row))

                # (Optional, but sometimes helps OpenCV)
                tiled = np.ascontiguousarray(tiled)

                # ---- Show in the same main thread ----
                cv2.imshow("LAB Multi-Camera Viewer", tiled)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()