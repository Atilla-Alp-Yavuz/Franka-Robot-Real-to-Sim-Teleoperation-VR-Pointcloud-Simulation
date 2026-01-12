# iris_receiver.py
import asyncio
import websockets
import json
import numpy as np
import cv2

async def receiver(ws):
    print("Client connected")
    try:
        while True:
            header_json = await ws.recv()
            header = json.loads(header_json)

            cam = header["camera"]
            w   = header["w"]
            h   = header["h"]
            pc_n = header.get("pc_points", 0)
            rgb_size = header["rgb_size"]
            depth_size = header["depth_size"]

            print(f"\n[RECEIVING {cam}]")
            print(" Expecting:", rgb_size, "bytes RGB,", depth_size, "bytes depth,", pc_n, "points")

            rgb_jpeg = await ws.recv()
            rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, np.uint8), cv2.IMREAD_COLOR)

            depth_bytes = await ws.recv()
            depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w)

            pc = None
            if pc_n > 0:
                pc_bytes = await ws.recv()
                pc = np.frombuffer(pc_bytes, dtype=np.float32).reshape(pc_n, 3)

            print(" DONE. shapes:")
            print("  RGB :", None if rgb is None else rgb.shape)
            print("  Depth:", depth.shape)
            if pc is not None:
                print("  PC  :", pc.shape)

    except websockets.ConnectionClosedOK:
        print("Client disconnected (clean close).")
    except websockets.ConnectionClosedError as e:
        print("Client disconnected (error):", e)

async def main():
    print("IRIS Receiver running at ws://127.0.0.1:8765")
    async with websockets.serve(receiver, "127.0.0.1", 8765, max_size=2**30):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
