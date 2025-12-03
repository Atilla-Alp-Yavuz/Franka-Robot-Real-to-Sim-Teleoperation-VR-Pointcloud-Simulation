# lab_live_viewer_headless.py
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from mujoco import viewer

# --------------- CONFIG ---------------
W, H = 320, 240
CAMERAS = ["top", "front", "left", "right"]
SHOW_DEPTH = False           # True → save depth visualization instead of RGB
FPS = 5                      # how often to save images (Hz)

LAB_XMLS = {
    0: "lab0.xml",
    1: "lab1_T_stack.xml",
    2: "lab2_boxes_cups.xml",
    3: "lab3_stick_maze.xml",
}


def load_lab(lab_id: int, W: int, H: int):
    xml_path = Path(LAB_XMLS[lab_id])
    print(f"[INFO] Loading lab {lab_id} from {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    return model, data


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Depth → BGR visualization for saving to disk."""
    depth = np.nan_to_num(depth, nan=5.0)
    dmin, dmax = depth.min(), depth.max()
    d = (depth - dmin) / (dmax - dmin + 1e-6)
    d = (255 * d).astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, data = load_lab(lab_id, W, H)

    with viewer.launch_passive(model, data) as v:
        with mujoco.Renderer(model, width=W, height=H) as renderer:

            last_save_time = 0.0
            frame_idx = 0

            print("[INFO] MuJoCo viewer running. Close it or Ctrl+C to stop.")
            while v.is_running():
                # --- your teleop / control could go here ---
                mujoco.mj_step(model, data)
                v.sync()

                now = time.time()
                if now - last_save_time < 1.0 / FPS:
                    continue
                last_save_time = now

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

                if len(frames) != 4:
                    print("[WARN] Expected 4 cameras, got", len(frames))
                    continue

                top_row = np.hstack((frames[0], frames[1]))
                bottom_row = np.hstack((frames[2], frames[3]))
                tiled = np.vstack((top_row, bottom_row))
                tiled = np.ascontiguousarray(tiled)

                out_name = f"lab{lab_id}_multi_{frame_idx:04d}.png"
                cv2.imwrite(out_name, tiled)
                print(f"[SAVE] {out_name}")
                frame_idx += 1