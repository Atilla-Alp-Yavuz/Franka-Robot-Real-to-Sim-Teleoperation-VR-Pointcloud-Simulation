import mujoco
import numpy as np
import zmq
import open3d as o3d
import time

# --- Configuration ---
# Camera Resolution (Lower this if lag occurs, e.g., 640x480)
WIDTH, HEIGHT = 640, 480 
FOVEA_ANGLE_DEG = 20.0  # Cone of high detail
PERIPHERAL_DOWNSAMPLE = 10 # Keep 1 in every 10 points outside fovea

class FoveatedBridge:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.context = zmq.Context()
        
        # 1. RECEIVE Gaze from Unity
        self.gaze_sub = self.context.socket(zmq.SUB)
        self.gaze_sub.connect("tcp://localhost:5556") # Connect to Unity
        self.gaze_sub.setsockopt_string(zmq.SUBSCRIBE, "gaze")
        
        # 2. PUBLISH Point Cloud to Unity
        self.pc_pub = self.context.socket(zmq.PUB)
        self.pc_pub.bind("tcp://*:5555")

        # Renderer Setup
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        
        # State
        self.user_pos = np.zeros(3)
        self.user_dir = np.array([0, 0, 1])

    def get_latest_gaze(self):
        """Non-blocking check for new gaze data"""
        try:
            while True: # Drain queue to get latest
                topic, msg = self.gaze_sub.recv_multipart(flags=zmq.NOBLOCK)
                # Parse "x,y,z,dx,dy,dz" string
                vals = [float(v) for v in msg.decode().split(',')]
                self.user_pos = np.array(vals[:3])
                self.user_dir = np.array(vals[3:])
        except zmq.Again:
            pass

    def process_camera(self, cam_name):
        # 1. Render Depth in MuJoCo
        self.renderer.update_scene(self.data, camera=cam_name)
        depth = self.renderer.render_depth()
        
        # 2. Fast Unprojection (using Open3D for C++ speed)
        # Note: You need to pre-calculate intrinsics based on MuJoCo fovy
        fovy = self.model.cam(cam_name).fovy[0]
        f = 0.5 * HEIGHT / np.tan(np.deg2rad(fovy) / 2)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, f, f, WIDTH/2, HEIGHT/2)
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth), intrinsic, depth_scale=1.0, depth_trunc=10.0, stride=4
        )
        
        # 3. Apply Foveation (The "Step 3" Logic)
        points = np.asarray(pcd.points)
        if points.shape[0] == 0: return

        # Transform points to world frame (using camera pose from MuJoCo)
        # ... (Matrix multiplication with data.cam_xmat/xpos) ...
        # For brevity, let's assume points are now in World Space
        
        # Calculate angle between (Point - Eye) and (GazeVector)
        to_point = points - self.user_pos
        to_point_norm = to_point / np.linalg.norm(to_point, axis=1, keepdims=True)
        dots = np.sum(to_point_norm * self.user_dir, axis=1)
        
        # Filter: High detail if dot product is close to 1 (looking at it)
        threshold = np.cos(np.deg2rad(FOVEA_ANGLE_DEG))
        mask_fovea = dots > threshold
        
        # High Res Fovea + Low Res Peripheral
        pts_fovea = points[mask_fovea]
        pts_periph = points[~mask_fovea][::PERIPHERAL_DOWNSAMPLE] # 10x decimation
        
        final_cloud = np.vstack((pts_fovea, pts_periph))
        
        # 4. Send to IRIS-Viz
        # Format: [Header: Int Count][Body: Floats]
        header = np.array([final_cloud.shape[0]], dtype=np.int32).tobytes()
        body = final_cloud.astype(np.float32).tobytes()
        self.pc_pub.send_multipart([b"pointcloud", header, body])

# Usage in your Main Loop
# bridge = FoveatedBridge(model, data)
# while sim_running:
#     bridge.get_latest_gaze()
#     bridge.process_camera("cam_front")
#     mujoco.mj_step(model, data)