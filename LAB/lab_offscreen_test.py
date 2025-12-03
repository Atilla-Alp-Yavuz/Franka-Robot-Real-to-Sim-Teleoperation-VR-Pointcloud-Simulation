# lab_offscreen_test.py
import sys
from pathlib import Path

import mujoco
import numpy as np
import cv2

CAMERA = "overhead"
W, H = 640, 480

LAB_XMLS = {
    0: "lab0.xml",
    1: "lab1_T_stack.xml",
    2: "lab2_boxes_cups.xml",
    3: "lab3_stick_maze.xml",
}

if __name__ == "__main__":
    lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    xml_path = Path(LAB_XMLS[lab_id])
    print(f"[INFO] Loading lab {lab_id} from {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("[INFO] Cameras in model:")
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  id={i}, name={name}")

    # IMPORTANT: set offscreen size *before* creating Renderer
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H

    with mujoco.Renderer(model, width=W, height=H) as renderer:
        print(f"[INFO] Rendering camera: {CAMERA}")
        renderer.update_scene(data, camera=CAMERA)
        rgb = renderer.render()

        print("[DEBUG] rgb dtype:", rgb.dtype, "shape:", rgb.shape)
        print("[DEBUG] rgb min/max:", float(rgb.min()), float(rgb.max()))
        print("[DEBUG] first 5 pixels:", rgb.reshape(-1, 3)[:5])

        out_name = f"lab{lab_id}_{CAMERA}_test.png"
        cv2.imwrite(out_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[SAVE] {out_name}")