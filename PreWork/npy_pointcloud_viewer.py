import numpy as np
import open3d as o3d

# -------------------------
# Load point cloud (.npy)
# -------------------------
pc = np.load("front_pc.npy")   # CHANGE filename

# Remove invalid points
mask = np.isfinite(pc).all(axis=1)
pc = pc[mask]

print("Loaded points:", pc.shape)

# -------------------------
# Center cloud for better visualization
# (world coords sometimes far from origin)
# -------------------------
pc_centered = pc - pc.mean(axis=0)

# -------------------------
# Convert to Open3D format
# -------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_centered)

# Optional: color the cloud for visibility
pcd.paint_uniform_color([1.0, 0.6, 0.2])  # light orange

# -------------------------
# High-quality visualization
# with zoom, front vector, lookat, and up direction
# -------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Franka Point Cloud", width=1280, height=720)
vis.add_geometry(pcd)

vc = vis.get_view_control()

# Zoom in aggressively so robot isn’t tiny
vc.set_zoom(0.5)

# Orient camera
vc.set_front([0.0, 0.0, -1.0])   # looking forward
vc.set_lookat([0.0, 0.0, 0.0])   # center of cloud
vc.set_up([0.0, -1.0, 0.0])      # camera up vector

vis.run()
vis.destroy_window()
