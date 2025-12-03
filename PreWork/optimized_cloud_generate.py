import mujoco
from pathlib import Path
import numpy as np
import cv2

# ----------------------------------
# Config
# ----------------------------------
W, H = 640, 480
CAMERAS = ["top", "front", "left", "right"]

# ----------------------------------
# Load MuJoCo model
# ----------------------------------
xml_path = Path("franka_emika_panda") / "mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Initial pose
initial_qpos = [0, -0.5, 0, -1.2, 0, 1.6, 0]
data.qpos[:7] = initial_qpos
mujoco.mj_forward(model, data)

model.vis.global_.offwidth = W
model.vis.global_.offheight = H


# ----------------------------------
# Helper: Compute camera intrinsics
# ----------------------------------
def get_intrinsics(model, cam_name, W, H):
    cam_id = model.cam(cam_name).id

    # MuJoCo 3.x: cam_fovy is already in RADIANS
    fovy = model.cam_fovy[cam_id]          # [rad]
    fy = H / (2.0 * np.tan(fovy / 2.0))
    fx = fy * (W / H)
    cx = W / 2.0
    cy = H / 2.0

    return fx, fy, cx, cy, cam_id


# ----------------------------------
# Back-project depth → camera-frame point cloud
# ----------------------------------
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Returns Nx3 point cloud in camera frame (only valid pixels)."""

    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]

    # Flatten
    zs = depth.reshape(-1)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    # Filter out invalid / background (z == 0)
    valid = zs > 0
    zs = zs[valid]
    xs = xs[valid]
    ys = ys[valid]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.vstack((X, Y, Z)).T  # (N, 3)


# ----------------------------------
# Transform camera-frame → world-frame
# ----------------------------------
def cam_to_world(model, data, cam_id, pts_cam):
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p = data.cam_xpos[cam_id]
    return pts_cam @ R.T + p


# ----------------------------------
# Main rendering
# ----------------------------------
with mujoco.Renderer(model, width=W, height=H) as renderer:
    for cam in CAMERAS:
        print(f"[Rendering]: {cam}")

        # ---------- RGB ----------
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        cv2.imwrite(f"{cam}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # ---------- Depth (GL → metric) ----------
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth_gl = renderer.render().copy()
        renderer.disable_depth_rendering()

        near = model.vis.map.znear
        far = model.vis.map.zfar
        depth = near * far / (far - (far - near) * depth_gl)

        # Background pixels sit exactly at 'far'
        depth[depth >= far - 1e-6] = 0.0

        # Save a visualization just to sanity-check
        depth_vis = (depth - depth[depth > 0].min()) / (
            depth[depth > 0].max() - depth[depth > 0].min() + 1e-6
        )
        depth_vis = (depth_vis * 255).astype(np.uint8)
        cv2.imwrite(f"{cam}_depth.png", depth_vis)

        # ---------- Point Cloud ----------
        fx, fy, cx, cy, cam_id = get_intrinsics(model, cam, W, H)
        pc_cam = depth_to_pointcloud(depth, fx, fy, cx, cy)
        pc_world = cam_to_world(model, data, cam_id, pc_cam)

        np.save(f"{cam}_pc.npy", pc_world)
        print(f" Saved {pc_world.shape[0]} 3D points → {cam}_pc.npy")

print("DONE — RGB, depth, and corrected point clouds saved.")
