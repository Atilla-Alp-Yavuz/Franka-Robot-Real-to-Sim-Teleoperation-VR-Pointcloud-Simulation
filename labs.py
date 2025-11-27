# labs.py
from pathlib import Path
import mujoco

LAB_XMLS = {
    0: "lab0.xml",
    1: "lab1_T_stack.xml",
    2: "lab2_boxes_cups.xml",
    3: "lab3_stick_maze.xml",
}

def load_lab(lab_id: int, offscreen_w: int | None = None, offscreen_h: int | None = None):
    xml_path = Path(LAB_XMLS[lab_id])
    print(f"[LAB] Loading lab {lab_id} from {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if offscreen_w is not None and offscreen_h is not None:
        model.vis.global_.offwidth = offscreen_w
        model.vis.global_.offheight = offscreen_h

    return model, data