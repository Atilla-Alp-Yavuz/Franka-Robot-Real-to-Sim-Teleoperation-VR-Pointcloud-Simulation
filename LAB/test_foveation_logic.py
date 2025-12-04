import numpy as np

FOVEA_ANGLE_DEG = 20.0
PERIPHERAL_DOWNSAMPLE = 10

def foveate(points, user_pos, user_dir):
    to_point = points - user_pos
    to_point_norm = to_point / np.linalg.norm(to_point, axis=1, keepdims=True)
    dots = np.sum(to_point_norm * user_dir, axis=1)

    threshold = np.cos(np.deg2rad(FOVEA_ANGLE_DEG))
    mask_fovea = dots > threshold

    pts_fovea = points[mask_fovea]
    pts_periph = points[~mask_fovea][::PERIPHERAL_DOWNSAMPLE]
    final_cloud = np.vstack((pts_fovea, pts_periph))
    return mask_fovea, final_cloud

if __name__ == "__main__":
    # Synthetic ring of points around origin at radius 1 in XY plane
    angles = np.linspace(-np.pi, np.pi, 200)
    points = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)

    user_pos = np.array([0.0, 0.0, 0.0])
    user_dir = np.array([1.0, 0.0, 0.0])  # looking along +X

    mask_fovea, final_cloud = foveate(points, user_pos, user_dir)

    fovea_count = mask_fovea.sum()
    print("Fovea count:", fovea_count)
    print("Total after foveation:", final_cloud.shape[0])

    # Sanity checks
    assert fovea_count > 0
    assert final_cloud.shape[0] < points.shape[0]  # we actually downsample
    print("Foveation logic ✅")
    #mjpython LAB/test_foveation_logic.py