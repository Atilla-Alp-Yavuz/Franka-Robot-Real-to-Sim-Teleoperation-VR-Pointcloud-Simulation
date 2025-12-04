using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;

public class FakePointcloudPublisher : MonoBehaviour
{
    public string topic = "pointcloud";
    public string port = "5555";

    private PublisherSocket _pub;
    private Thread _thread;
    private bool _running = false;

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _pub = new PublisherSocket();
        _pub.Bind($"tcp://*:{port}");

        _running = true;
        _thread = new Thread(PublishLoop);
        _thread.IsBackground = true;
        _thread.Start();
    }

    private void PublishLoop()
    {
        // Let subscribers connect
        Thread.Sleep(200);

        System.Random rng = new System.Random();

        while (_running)
        {
            int N = 1000;
            float[] floats = new float[N * 3];

            // Random sphere
            for (int i = 0; i < N; i++)
            {
                float theta = (float)(2 * Math.PI * rng.NextDouble());
                float phi = (float)(Math.PI * rng.NextDouble());
                float r = 1.0f;

                float x = r * Mathf.Sin(phi) * Mathf.Cos(theta);
                float y = r * Mathf.Sin(phi) * Mathf.Sin(theta);
                float z = r * Mathf.Cos(phi);

                int idx = i * 3;
                floats[idx] = x;
                floats[idx + 1] = y;
                floats[idx + 2] = z;
            }

            byte[] header = BitConverter.GetBytes(N);
            byte[] body = new byte[floats.Length * 4];
            Buffer.BlockCopy(floats, 0, body, 0, body.Length);

            _pub.SendMoreFrame(topic).SendMoreFrame(header).SendFrame(body);

            Thread.Sleep(50); // ~20 Hz
        }
    }

    void OnDestroy()
    {
        _running = false;
        _thread?.Join();
        _pub?.Close();
        NetMQConfig.Cleanup();
    }
}