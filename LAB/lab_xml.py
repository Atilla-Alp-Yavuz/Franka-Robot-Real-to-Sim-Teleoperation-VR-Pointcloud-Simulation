# LAB/show_xml.py
import sys
from pathlib import Path
import mujoco
from mujoco import viewer

LAB_XMLS = {
    0: "lab0.xml",
    1: "lab1_T_stack.xml",
    2: "lab2_boxes_cups.xml",
    3: "lab3_stick_maze.xml",
}

HERE = Path(__file__).resolve().parent  # .../LAB

def load_model(lab_id: int):
    xml_path = (HERE / LAB_XMLS[lab_id]).resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    print("[MuJoCo] Loading:", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data

if __name__ == "__main__":
    lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, data = load_model(lab_id)

    # Interactive window (blocks until you close it)
    viewer.launch(model, data)
    #mjpython lab_xml.py 0
    #mjpython lab_xml.py 0

    