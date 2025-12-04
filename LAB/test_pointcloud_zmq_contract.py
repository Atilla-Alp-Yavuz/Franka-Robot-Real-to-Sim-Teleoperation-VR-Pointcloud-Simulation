import zmq
import numpy as np
import threading
import time

TOPIC = b"pointcloud"
PORT = 5555

def publisher():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{PORT}")
    time.sleep(0.2)  # give sub time to connect

    N = 10
    points = np.arange(N * 3, dtype=np.float32).reshape(-1, 3)
    header = np.array([N], dtype=np.int32).tobytes()
    body = points.astype(np.float32).tobytes()

    pub.send_multipart([TOPIC, header, body])
    print("[PUB] Sent", N, "points")
    time.sleep(0.5)
    pub.close()

def subscriber():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{PORT}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "pointcloud")

    topic, header, body = sub.recv_multipart()
    assert topic == TOPIC

    N = int(np.frombuffer(header, dtype=np.int32)[0])
    assert N == 10, f"Expected 10 points, got {N}"

    floats = np.frombuffer(body, dtype=np.float32)
    assert floats.shape[0] == N * 3, f"Expected {N*3} floats"
    pts = floats.reshape(-1, 3)

    print("[SUB] Received shape:", pts.shape)
    print("[SUB] First point:", pts[0])

    sub.close()

if __name__ == "__main__":
    t_pub = threading.Thread(target=publisher)
    t_sub = threading.Thread(target=subscriber)

    t_sub.start()
    t_pub.start()

    t_pub.join()
    t_sub.join()
    print("ZMQ pointcloud contract ✅")