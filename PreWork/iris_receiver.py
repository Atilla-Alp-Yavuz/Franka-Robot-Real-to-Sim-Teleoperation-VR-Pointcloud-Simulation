import asyncio
import websockets
import json
import numpy as np
import cv2

async def receiver(ws):
    print("Client connected")

    while True:
        # 1. Receive header
        header_json = await ws.recv()
        header = json.loads(header_json)

        cam = header["camera"]
        w   = header["w"]
        h   = header["h"]
        pc_n = header["pc_points"]
        rgb_size = header["rgb_size"]
        depth_size = header["depth_size"]

        print(f"\n[RECEIVING {cam}]")
        print(" Expecting:", rgb_size, "bytes RGB,", depth_size, "bytes depth,", pc_n, "points")

        # 2. Receive RGB JPEG
        rgb_jpeg = await ws.recv()
        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, np.uint8), cv2.IMREAD_COLOR)

        # 3. Receive Depth
        depth_bytes = await ws.recv()
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w)

        # 4. Receive Pointcloud
        pc_bytes = await ws.recv()
        pc = np.frombuffer(pc_bytes, dtype=np.float32).reshape(pc_n, 3)

        print(" DONE. shapes:")
        print("  RGB:", rgb.shape)
        print("  Depth:", depth.shape)
        print("  PointCloud:", pc.shape)

async def main():
    print("IRIS Receiver running at ws://127.0.0.1:8765")
    async with websockets.serve(
        lambda ws, path: receiver(ws),
        "127.0.0.1",
        8765,
        max_size=2**30
    ):
        await asyncio.Future()

asyncio.run(main())
