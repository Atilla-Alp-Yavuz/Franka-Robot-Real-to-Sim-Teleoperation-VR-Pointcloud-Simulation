# vision_utils.py
import numpy as np
import mujoco

def get_intrinsics(model, cam_name, W, H):
    cam_id = model.cam(cam_name).id
    # MuJoCo 3.x: cam_fovy is in radians
    fovy = model.cam_fovy[cam_id]
    fy = H / (2.0 * np.tan(fovy / 2.0))
    fx = fy * (W / H)
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy, cam_id

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]

    zs = depth.reshape(-1)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    valid = zs > 0
    zs = zs[valid]
    xs = xs[valid]
    ys = ys[valid]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.vstack((X, Y, Z)).T  # (N, 3)

def cam_to_world(model, data, cam_id, pts_cam):
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p = data.cam_xpos[cam_id]
    return pts_cam @ R.T + p