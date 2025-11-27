# teleop_interface.py
import socket
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TeleopCommand:
    q: Optional[np.ndarray] = field(default=None)    # shape (7,)
    gripper: Optional[float] = field(default=None)   # scalar in [0,1] or width


class TeleopSource:
    """
    Simple TCP-based teleop source.
    - Listens on 127.0.0.1:5000
    - Expects lines: q1 q2 q3 q4 q5 q6 q7 gripper
      (space-separated, all floats)
    - Keeps the *latest* parsed command for the main loop to read.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self._latest_cmd = TeleopCommand()
        self._lock = threading.Lock()
        self._stop = False

        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()
        print(f"[TELEOP] Listening for teleop client on {host}:{port}")

    def _server_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(1.0)

        try:
            while not self._stop:
                try:
                    print("[TELEOP] Waiting for client connection...")
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue

                print(f"[TELEOP] Client connected: {addr}")
                with conn:
                    conn_file = conn.makefile("r")
                    for line in conn_file:
                        if self._stop:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 8:
                            print(f"[TELEOP] Invalid line (expected 8 floats): {line}")
                            continue
                        try:
                            values = [float(x) for x in parts]
                        except ValueError:
                            print(f"[TELEOP] Could not parse floats: {line}")
                            continue

                        q = np.array(values[:7], dtype=float)
                        grip = float(values[7])
                        grip = max(0.0, min(1.0, grip))  # clamp to [0,1]

                        with self._lock:
                            self._latest_cmd = TeleopCommand(q=q, gripper=grip)
                print("[TELEOP] Client disconnected.")
        finally:
            sock.close()

    def update(self) -> TeleopCommand:
        """
        Called from main loop. Returns latest command (copy).
        """
        with self._lock:
            cmd = self._latest_cmd
            # return a shallow copy to avoid external mutation
            return TeleopCommand(
                q=None if cmd.q is None else cmd.q.copy(),
                gripper=cmd.gripper
            )

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)