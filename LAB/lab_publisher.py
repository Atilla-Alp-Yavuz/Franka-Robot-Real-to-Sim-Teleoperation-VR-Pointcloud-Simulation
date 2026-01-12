import time
import mujoco
from simpub.sim.mj_publisher import MujocoPublisher

XML_PATH = r"C:\Users\AtillaGokay\Desktop\Franka-Robot-Real-to-Sim-Teleoperation-VR-Pointcloud-Simulation\LAB\lab1_T_stack.xml"
HOST = "192.168.0.208"
NAMES = mujoco.mjtObj.mjOBJ_ALL # "IRLMQ3-1"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
name = model.names(NAMES)

publisher = MujocoPublisher(
    model,
    data,
    host=HOST,
    visible_geoms_groups=list(range(10))  # show all
)

# one step so xpos/xquat are valid
mujoco.mj_step(model, data)

print("Lab publisher running...")
while True:
    mujoco.mj_step(model, data)
    time.sleep(1 / 60)
