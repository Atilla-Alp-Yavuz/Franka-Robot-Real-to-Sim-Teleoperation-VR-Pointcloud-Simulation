import sys
from pathlib import Path
import mujoco
from mujoco import viewer

# Choose lab: 0, 1, 2, or 3 (e.g. from CLI: mjpython lab_test.py 2)
lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

xml_map = {
    0: "lab0.xml",                 # just robot + table
    1: "T_lab.xml",
    2: "box_lab.xml",
    3: "stick_maze.xml",
}

xml_path = Path(xml_map[lab_id])
print(f"Loading {xml_path}")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)