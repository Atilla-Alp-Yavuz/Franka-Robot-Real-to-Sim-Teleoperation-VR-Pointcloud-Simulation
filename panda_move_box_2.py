import time
from pathlib import Path

import mujoco
from mujoco import viewer
import numpy as np

XML_PATH = Path("panda_scene.xml")

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
FINGER_JOINT = "finger_joint1"
BOX_GEOM_NAME = "box_geom"
GRIPPER_SITE_NAME = "gripper"


def joint_qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[j_id]


def joint_dof_index(model: mujoco.MjModel, joint_name: str) -> int:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_dofadr[j_id]


def get_site_world_pos(model, data, site_name: str) -> np.ndarray:
    s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[s_id].copy()


def interpolate(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * q_start + alpha * q_end


### NEW: small IK step to move gripper forward in +x
def nudge_gripper_forward(model, data, arm_joint_names, dx_forward: float):
    """
    Move gripper approximately dx_forward in +x (base frame),
    adjusting only the 7 arm joints via a Jacobian pseudo-inverse step.
    """
    # 1) Compute Jacobian at gripper site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE_NAME)
    J_pos = np.zeros((3, model.nv))   # position Jacobian
    mujoco.mj_jacSite(model, data, J_pos, None, site_id)

    # 2) Extract columns corresponding to the 7 arm joints (their dof indices)
    arm_dofs = np.array([joint_dof_index(model, name) for name in arm_joint_names], dtype=int)
    J_arm = J_pos[:, arm_dofs]   # shape (3,7)

    # 3) Desired Cartesian motion: small step in +x
    dx = np.array([dx_forward, 0.0, 0.0])  # [dx, dy, dz]

    # 4) Solve least-squares: J_arm * dq ≈ dx
    dq_arm = np.linalg.pinv(J_arm) @ dx   # shape (7,)

    # 5) Apply dq to arm joints (qpos and dof indices are aligned for hinged joints)
    for i, name in enumerate(arm_joint_names):
        qadr = joint_qpos_index(model, name)
        data.qpos[qadr] += dq_arm[i]

    # Recompute forward kinematics
    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    arm_idx = np.array([joint_qpos_index(model, j) for j in ARM_JOINTS], dtype=int)
    finger_idx = joint_qpos_index(model, FINGER_JOINT)

    GRIPPER_OPEN = 0.04
    GRIPPER_CLOSED = 0.0

    # ---------------- NEW q_grasp from you ---------------- #
    q_home = np.array([0.0, -0.5, 0.0, -1.2, 0.0, 1.6, 0.0])

    q_pre_grasp = np.array([  # can keep similar to before
        0.0,
       -0.2,
        0.0,
       -1.8,
        0.0,
        1.9,
        0.0
    ])

    q_grasp = np.array([
        0.0,
        0.3,
        0.0,
       -2.45,
        0.2,
        2.6,
        0.35
    ])

    q_lift  = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 2.0, 0.0])
    q_right = np.array([0.5, -0.3, 0.0, -1.8, 0.0, 2.0, 0.0])
    q_place = np.array([0.5,  0.3, 0.0, -2.45, 0.2, 2.6, 0.0])

    # -------------- Put robot in your q_grasp -------------- #
    data.qpos[arm_idx] = q_grasp
    data.qpos[finger_idx] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    # Debug: see current position
    grip_before = get_site_world_pos(model, data, GRIPPER_SITE_NAME)
    print("Gripper before nudge:", grip_before)

    # -------------- Nudge gripper forward by e.g. 3 cm ----- #
    nudge_gripper_forward(model, data, ARM_JOINTS, dx_forward=0.1)

    grip_after = get_site_world_pos(model, data, GRIPPER_SITE_NAME)
    print("Gripper after nudge:", grip_after)
    print("Delta x ≈", grip_after[0] - grip_before[0])

    # Save the new q_grasp (so you can hard-code it if you like)
    new_q_grasp = data.qpos[arm_idx].copy()
    print("New q_grasp to paste into code:", new_q_grasp)

    # Now reset the timeline poses using this updated q_grasp
    q_grasp = new_q_grasp

    # Reset to home to start animation
    data.qpos[arm_idx] = q_home
    data.qpos[finger_idx] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    # Phases as before, but using updated q_grasp
    phases = [
        (0.0, 3.0, q_home,      q_pre_grasp, GRIPPER_OPEN,   GRIPPER_OPEN),
        (3.0, 6.0, q_pre_grasp, q_grasp,     GRIPPER_OPEN,   GRIPPER_OPEN),
        (6.0, 8.0, q_grasp,     q_grasp,     GRIPPER_OPEN,   GRIPPER_CLOSED),
        (8.0, 11.0, q_grasp,    q_lift,      GRIPPER_CLOSED, GRIPPER_CLOSED),
        (11.0, 14.0, q_lift,    q_right,     GRIPPER_CLOSED, GRIPPER_CLOSED),
        (14.0, 17.0, q_right,   q_place,     GRIPPER_CLOSED, GRIPPER_CLOSED),
        (17.0, 19.0, q_place,   q_place,     GRIPPER_CLOSED, GRIPPER_OPEN),
        (19.0, 22.0, q_place,   q_home,      GRIPPER_OPEN,   GRIPPER_OPEN),
    ]

    with viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running():
            t = time.time() - t0

            if t > phases[-1][1]:
                t0 = time.time()
                continue

            for (t_start, t_end, q_from, q_to, g_from, g_to) in phases:
                if t_start <= t <= t_end:
                    alpha = (t - t_start) / (t_end - t_start)
                    q_target = interpolate(q_from, q_to, alpha)
                    g_target = (1.0 - alpha) * g_from + alpha * g_to

                    data.qpos[arm_idx] = q_target
                    data.qpos[finger_idx] = g_target
                    break

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()