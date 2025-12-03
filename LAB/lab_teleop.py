# lab_teleop.py
import sys
import time

import mujoco
from mujoco import viewer
import numpy as np

from LAB.labs import load_lab
from PreWork.teleop_interface import TeleopSource
from franka_real import FrankaReal
from LAB.sim_control import apply_joint_command_to_sim, apply_gripper_to_sim

CONTROL_HZ = 250.0      # control frequency
SYNC_REAL = True        # set False to disable real-robot logging

if __name__ == "__main__":
    lab_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # 1) Load lab model
    model, data = load_lab(lab_id)

    # 2) Teleop source (TCP listener)
    teleop = TeleopSource(host="127.0.0.1", port=5000)

    # 3) Real robot wrapper (stub)
    real_robot = FrankaReal() if SYNC_REAL else None

    dt = 1.0 / CONTROL_HZ
    last_time = time.time()

    try:
        with viewer.launch_passive(model, data) as v:
            print(f"[TELEOP] Running lab {lab_id} at {CONTROL_HZ} Hz.")
            print("[TELEOP] Close the MuJoCo window or Ctrl+C to stop.")
            step_count = 0

            while v.is_running():
                now = time.time()
                if now - last_time < dt:
                    time.sleep(0.0005)
                    continue
                last_time = now
                step_count += 1

                # --- 1) Get latest teleop command ---
                cmd = teleop.update()
                q_cmd = cmd.q
                grip_cmd = cmd.gripper

                # --- 2) Apply to SIM ---
                if q_cmd is not None:
                    apply_joint_command_to_sim(model, data, q_cmd)
                if grip_cmd is not None:
                    apply_gripper_to_sim(model, data, grip_cmd)

                # --- 3) Apply to REAL (stub logging) ---
                if real_robot is not None and q_cmd is not None:
                    real_robot.send_joint_command(q_cmd)
                if real_robot is not None and grip_cmd is not None:
                    real_robot.send_gripper_command(grip_cmd)

                # --- 4) Step simulation ---
                mujoco.mj_step(model, data)

                # --- 5) Update viewer ---
                v.sync()
    finally:
        teleop.stop()