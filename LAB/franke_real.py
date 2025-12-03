# franka_real.py
import numpy as np

class FrankaReal:
    """
    Stub wrapper for the real Franka.
    Currently just logs commands.
    Replace internal methods with your actual robot API when ready.
    """

    def __init__(self):
        print("[REAL] FrankaReal initialized (stub, no hardware control).")

    def send_joint_command(self, q_cmd: np.ndarray):
        # Here you'd talk to real Franka, e.g. via your existing library.
        # For now we just log every ~50th call to avoid spam.
        if hasattr(self, "_counter"):
            self._counter += 1
        else:
            self._counter = 0
        if self._counter % 50 == 0:
            print(f"[REAL] Joint command (stub): {q_cmd}")

    def send_gripper_command(self, gripper: float):
        if hasattr(self, "_gcounter"):
            self._gcounter += 1
        else:
            self._gcounter = 0
        if self._gcounter % 50 == 0:
            print(f"[REAL] Gripper command (stub): {gripper:.3f}")