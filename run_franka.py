import mujoco
from mujoco import viewer
from pathlib import Path
import time
import math

# Path to the Franka XML inside the project
xml_path = Path("panda_scene.xml") 

# Load model & data
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)       # <-- THIS WAS MISSING

# Put the arm in a nicer initial pose (optional)
initial_qpos = [0.0, -0.1, 0.0, -1.2, 0.0, 1.6, 0.0]
data.qpos[:len(initial_qpos)] = initial_qpos
mujoco.mj_forward(model, data)

# Launch an interactive viewer
with viewer.launch_passive(model, data) as v:
    t0 = time.time()
    while v.is_running():
        t = time.time() - t0

        # Simple demo: oscillate joint 1 and 2
        data.qpos[0] = initial_qpos[0] + 0.3 * math.sin(0.5 * t)
        data.qpos[1] = initial_qpos[1] + 0.2 * math.sin(0.8 * t)

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()

