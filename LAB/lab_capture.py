# lab_capture.py
import sys
from pathlib import Path
import mujoco
import numpy as np
import cv2

from PreWork.vision_utils import get_intrinsics, depth_to_pointcloud, cam_to_world

W, H = 640, 480
CAMERAS = ["top", "front", "left", "right"]

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

    # If you want an initial pose, set it here:
    # data.qpos[:7] = [...]
    mujoco.mj_forward(model, data)

    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    return model, data

if __name__ == "__main__":
    lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, data = load_lab(lab_id, W, H)
    print(f"Loaded lab {lab_id}: {LAB_XMLS[lab_id]}")

    with mujoco.Renderer(model, width=W, height=H) as renderer:
        for cam in CAMERAS:
            print(f"[Rendering]: {cam}")

            # ----- RGB -----
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            cv2.imwrite(f"lab{lab_id}_{cam}_rgb.png",
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # ----- Depth (GL → metric) -----
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=cam)
            depth_gl = renderer.render().copy()
            renderer.disable_depth_rendering()

            near = model.vis.map.znear
            far = model.vis.map.zfar
            depth = near * far / (far - (far - near) * depth_gl)

            # Background at far plane → 0
            depth[depth >= far - 1e-6] = 0.0

            # Save depth visualization (only valid pixels)
            valid = depth > 0
            if valid.any():
                dmin, dmax = depth[valid].min(), depth[valid].max()
                depth_vis = (depth - dmin) / (dmax - dmin + 1e-6)
            else:
                depth_vis = depth
            depth_vis = (depth_vis * 255).astype(np.uint8)
            cv2.imwrite(f"lab{lab_id}_{cam}_depth.png", depth_vis)

            # ----- Point cloud -----
            fx, fy, cx, cy, cam_id = get_intrinsics(model, cam, W, H)
            pc_cam = depth_to_pointcloud(depth, fx, fy, cx, cy)
            pc_world = cam_to_world(model, data, cam_id, pc_cam)

            np.save(f"lab{lab_id}_{cam}_pc.npy", pc_world)
            print(f" Saved {pc_world.shape[0]} points → lab{lab_id}_{cam}_pc.npy")

    print("DONE.")