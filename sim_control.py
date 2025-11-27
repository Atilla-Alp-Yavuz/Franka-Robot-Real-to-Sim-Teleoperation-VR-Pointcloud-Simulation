# sim_control.py
import numpy as np
import mujoco

def apply_joint_command_to_sim(model, data, q_cmd: np.ndarray):
    """
    Direct joint position control via actuators.
    Assumes:
    - 7 actuators for joints 1..7 in data.ctrl[0..6]
    - actuator gain/bias already set in XML for position control.
    """
    if q_cmd.shape[0] != 7:
        raise ValueError(f"q_cmd must be shape (7,), got {q_cmd.shape}")
    data.ctrl[:7] = q_cmd

def apply_gripper_to_sim(model, data, gripper: float):
    """
    Map gripper command to finger width.
    Assumes:
    - actuator8 (index 7) controls finger_joint1 with ctrlrange 0..0.04 (meters).
    - equality in XML couples finger_joint1 & finger_joint2.
    """
    gripper = float(gripper)
    gripper = max(0.0, min(1.0, gripper))  # clamp
    max_width = 0.04  # meters (from your XML range)
    width = gripper * max_width
    data.ctrl[7] = width