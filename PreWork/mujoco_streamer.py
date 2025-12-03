import asyncio
import websockets
import json
import numpy as np
import cv2
import mujoco
from pathlib import Path

# ------------------------------------
# CONFIG
# ------------------------------------
SERVER_URI = "ws://127.0.0.1:8765"
W, H = 640, 480
CAMERAS = ["front", "left", "right", "top"]

# ------------------------------------
# Load MuJoCo model
# ------------------------------------
xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

# Ensure offscreen buffer is large enough
model.vis.global_.offwidth = W
model.vis.global_.offheight = H


# ------------------------------------
# Camera intrinsics
# ------------------------------------
def get_intrinsics(model, cam_name, W, H):
    cam_id = model.cam(cam_name).id
    fovy = model.cam_fovy[cam_id] * np.pi / 180.0

    fy = H / (2 * np.tan(fovy / 2))
    fx = fy * (W / H)

    cx = W / 2
    cy = H / 2

    return fx, fy, cx, cy, cam_id


# ------------------------------------
# Depth → camera-frame point cloud
# ------------------------------------
def depth_to_pc(depth, fx, fy, cx, cy):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]

    Z = depth.flatten()
    X = (xs.flatten() - cx) * Z / fx
    Y = (ys.flatten() - cy) * Z / fy

    return np.stack((X, Y, Z), axis=1)  # Nx3


# ------------------------------------
# Transform camera-frame → world-frame
# ------------------------------------
def cam_to_world(model, data, cam_id, pts):
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p = data.cam_xpos[cam_id]
    return pts @ R.T + p  # Nx3


# ------------------------------------
# Streaming loop
# ------------------------------------
async def stream_loop():

    async with websockets.connect(SERVER_URI, max_size=None) as ws:
        print("Connected to receiver:", SERVER_URI)

        with mujoco.Renderer(model, width=W, height=H) as renderer:

            while True:
                mujoco.mj_step(model, data)

                for cam in CAMERAS:

                    # ------------------------------
                    # 1. RGB
                    # ------------------------------
                    renderer.update_scene(data, camera=cam)
                    rgb = renderer.render()
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    rgb_bytes = cv2.imencode(".jpg", rgb_bgr)[1].tobytes()

                    # ------------------------------
                    # 2. DEPTH
                    # ------------------------------
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=cam)
                    depth = renderer.render()
                    renderer.disable_depth_rendering()

                    # ------------------------------
                    # 3. POINT CLOUD
                    # ------------------------------
                    fx, fy, cx, cy, cam_id = get_intrinsics(model, cam, W, H)
                    pc_cam = depth_to_pc(depth, fx, fy, cx, cy)
                    pc_world = cam_to_world(model, data, cam_id, pc_cam)

                    # ------------------------------
                    # 4. HEADER (metadata)
                    # ------------------------------
                    header = {
                        "camera": cam,
                        "w": W,
                        "h": H,
                        "rgb_size": len(rgb_bytes),
                        "depth_size": depth.size * 4,     # float32
                        "pc_points": pc_world.shape[0]
                    }

                    # ------------------------------
                    # 5. SEND DATA IN ORDER
                    # ------------------------------
                    await ws.send(json.dumps(header))                     # HEADER
                    await ws.send(rgb_bytes)                               # RGB
                    await ws.send(depth.astype(np.float32).tobytes())      # DEPTH
                    await ws.send(pc_world.astype(np.float32).tobytes())   # POINT CLOUD

                # 20 FPS
                await asyncio.sleep(0.05)


# ------------------------------------
# RUN
# ------------------------------------
asyncio.run(stream_loop())
