# LAB/run_foveated_bridge.py
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from mujoco_iris_bridge import FoveatedBridge  # your class

XML = Path("LAB/lab0.xml")  # adjust if needed


def main():
    os.environ.setdefault("MUJOCO_GL", "glfw")

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    bridge = FoveatedBridge(model, data)

    dt = 0.01  # 100 Hz sim; you can tweak this

    print("[Bridge] Starting main loop. Press Ctrl-C to stop.")
    try:
        while True:
            # Step the MuJoCo simulation
            mujoco.mj_step(model, data)

            # Get latest gaze from Unity (if any)
            bridge.get_latest_gaze()

            # Render + foveate + send pointcloud
            bridge.process_camera("overhead")  # or your camera name

            # Simple real-time-ish pacing
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user (Ctrl-C).")

    finally:
        print("[Bridge] Shutting down cleanly.")


if __name__ == "__main__":
    main()