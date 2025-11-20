import time
from pathlib import Path

import mujoco
from mujoco import viewer
import numpy as np


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

XML_PATH = Path("panda_scene.xml")

# Joint names for the Panda arm in your XML
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# Slide joint for gripper (second one is coupled via equality in XML)
FINGER_JOINT = "finger_joint1"

# Geom and site names
BOX_GEOM_NAME = "box_geom"
GRIPPER_SITE_NAME = "gripper"


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def joint_qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the starting qpos index for a given joint name."""
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[j_id]


def get_site_world_pos(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    """Return the world position (x,y,z) of a site."""
    s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[s_id].copy()


def get_box_center_world_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the world position of the box body center."""
    g_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM_NAME)
    body_id = model.geom_bodyid[g_id]
    return data.xpos[body_id].copy()


def interpolate(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation between two joint configs."""
    return (1.0 - alpha) * q_start + alpha * q_end


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    # Load model + data
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Build joint index arrays
    arm_idx = np.array([joint_qpos_index(model, j) for j in ARM_JOINTS], dtype=int)
    finger_idx = joint_qpos_index(model, FINGER_JOINT)

    # Gripper open/close (slide joint range is [0, 0.04])
    GRIPPER_OPEN = 0.04
    GRIPPER_CLOSED = 0.0

    # --------------------------- Joint Poses -------------------------------- #

    # Home pose (nice viewing pose)
    q_home = np.array([0.0, -0.5, 0.0, -1.2, 0.0, 1.6, 0.0])

    # These poses are approximate and may need a little tuning:
    #  - q_pre_grasp: above the box
    #  - q_grasp:     down near the floor to actually grab the box
    #  - q_lift:      lifted after grasp
    #  - q_right:     moved to the right
    #  - q_place:     lower again to place
# Fully open pose above the box (centered, straight approach)
    q_pre_grasp = np.array([
        0.0,     # joint1 - no sideways rotation
    -0.5,     # joint2 - bring upper arm forward
        0.0,     
    -1.5,     # joint3 - elbow downwards
        0.0,
        1.8,
        0.0
    ])

    # Lower the hand straight down
    q_grasp = np.array([
        0.0,      # joint1 – front
        0.3,      # joint2 – forward
        0.0,      # joint3 – neutral about yaw
    -2.45,      # joint4 – elbow down
        0.2,      # joint5 – minimal roll
        2.6,      # joint6 – wrist up (so that hand is pointing down)
        0.0       # joint7 – tool orientation
    ])

    # Lift back up (no angle change)
    q_lift = np.array([
        0.0,
    -0.5,
        0.0,
    -1.5,
        0.0,
        1.8,
        0.0
    ])

    # Move to right
    q_right = np.array([
        1.0,     # rotate base right
    -0.5,
        0.0,
    -1.5,
        0.0,
        1.8,
        0.0
    ])

    # Lower to place
    q_place = np.array([
        0.4,
    -0.2,
        0.0,
    -2.0,
        0.0,
        2.0,
        0.0
    ])

    # -------------------- Initialize configuration ------------------------- #

    # Start from home pose with gripper open
    data.qpos[arm_idx] = q_home
    data.qpos[finger_idx] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    # --- Debug: print z of box vs. gripper at grasp pose ------------------- #
    # This is to help you tune q_grasp. It runs once before simulation.

    # Get box info at its initial XML-defined position
    box_center = get_box_center_world_pos(model, data)
    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM_NAME)
    box_half_z = model.geom_size[box_geom_id][2]
    box_top_z = box_center[2] + box_half_z

    # Temporarily set q_grasp to inspect where the gripper will be
    data.qpos[arm_idx] = q_grasp
    data.qpos[finger_idx] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    grip_pos = get_site_world_pos(model, data, GRIPPER_SITE_NAME)

    print("------ Debug pose info ------")
    print(f"Box center z: {box_center[2]:.4f}")
    print(f"Box top z:    {box_top_z:.4f}")
    print(f"Gripper z @ q_grasp: {grip_pos[2]:.4f}")
    print("(Ideally gripper z is close to box_top_z for a clean grasp.)")
    print("gripper z @ q_grasp:", grip_pos[2])
    print("-----------------------------")

    # Put arm back to home before running the animation
    data.qpos[arm_idx] = q_home
    data.qpos[finger_idx] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    # -------------------- Define pick & place phases ----------------------- #

    # Each phase: (start_t, end_t, q_from, q_to, g_from, g_to)
    phases = [
        # Move from home -> pre-grasp (open gripper)
        (0.0, 3.0, q_home,      q_pre_grasp, GRIPPER_OPEN,   GRIPPER_OPEN),
        # Move down towards box (still open)
        (3.0, 6.0, q_pre_grasp, q_grasp,     GRIPPER_OPEN,   GRIPPER_OPEN),
        # Close gripper at grasp pose
        (6.0, 8.0, q_grasp,     q_grasp,     GRIPPER_OPEN,   GRIPPER_CLOSED),
        # Lift with box
        (8.0, 11.0, q_grasp,    q_lift,      GRIPPER_CLOSED, GRIPPER_CLOSED),
        # Move to the right
        (11.0, 14.0, q_lift,    q_right,     GRIPPER_CLOSED, GRIPPER_CLOSED),
        # Lower to place pose
        (14.0, 17.0, q_right,   q_place,     GRIPPER_CLOSED, GRIPPER_CLOSED),
        # Open gripper to release
        (17.0, 19.0, q_place,   q_place,     GRIPPER_CLOSED, GRIPPER_OPEN),
        # Go back home
        (19.0, 22.0, q_place,   q_home,      GRIPPER_OPEN,   GRIPPER_OPEN),
    ]

    # ----------------------------------------------------------------------- #
    #  Simulation loop
    # ----------------------------------------------------------------------- #

    with viewer.launch_passive(model, data) as v:
        t0 = time.time()

        while v.is_running():
            t = time.time() - t0

            # Loop animation
            if t > phases[-1][1]:
                t0 = time.time()
                continue

            # Find active phase
            for (t_start, t_end, q_from, q_to, g_from, g_to) in phases:
                if t_start <= t <= t_end:
                    alpha = (t - t_start) / (t_end - t_start)
                    q_target = interpolate(q_from, q_to, alpha)
                    g_target = (1.0 - alpha) * g_from + alpha * g_to

                    # Apply targets directly to qpos
                    data.qpos[arm_idx] = q_target
                    data.qpos[finger_idx] = g_target
                    break

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()