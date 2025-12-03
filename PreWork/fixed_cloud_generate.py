import mujoco
from pathlib import Path
import numpy as np
import cv2

W, H = 640, 480
CAMERAS = ["top", "front", "left", "right"]

xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

model.vis.global_.offwidth = W
model.vis.global_.offheight = H

def get_intrinsics(model, cam_name, W, H):
    cam_id = model.cam(cam_name).id
    fovy = model.cam_fovy[cam_id] * np.pi / 180.0
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy * (W / H)
    cx = W / 2
    cy = H / 2
    return fx, fy, cx, cy, cam_id

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]

    zs = depth.flatten()
    xs = xs.flatten()
    ys = ys.flatten()

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.vstack((X, Y, Z)).T

def cam_to_world(model, data, cam_id, pts_cam):
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p = data.cam_xpos[cam_id]
    return pts_cam @ R.T + p

with mujoco.Renderer(model, width=W, height=H) as renderer:
    for cam in CAMERAS:
        print(f"[Rendering]: {cam}")

        # RGB
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        cv2.imwrite(f"{cam}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # DEPTH (GL → METRIC conversion)
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth_gl = renderer.render().copy()
        renderer.disable_depth_rendering()

        # Convert GL depth to REAL depth (meters)
        near = model.vis.map.znear
        far = model.vis.map.zfar
        depth = near * far / (far - (far - near) * depth_gl)

        # Visualize depth
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        cv2.imwrite(f"{cam}_depth.png", depth_vis)

        # POINT CLOUD
        fx, fy, cx, cy, cam_id = get_intrinsics(model, cam, W, H)
        pc_cam = depth_to_pointcloud(depth, fx, fy, cx, cy)
        pc_world = cam_to_world(model, data, cam_id, pc_cam)

        np.save(f"{cam}_pc.npy", pc_world)
        print(f"Saved {pc_world.shape[0]} points → {cam}_pc.npy")

print("DONE.")
