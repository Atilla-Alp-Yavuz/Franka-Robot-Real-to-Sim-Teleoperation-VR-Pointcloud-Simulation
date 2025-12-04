using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Threading;
using System;
using System.Collections.Generic;

public class FoveatedPointcloudSubscriber : MonoBehaviour
{
    private SubscriberSocket _subSocket;
    private Thread _listenerThread;
    private bool _running = false;

    public string connectToPort = "5555";
    public string topic = "pointcloud";
    
    public event Action<Vector3[]> OnPointcloudReceived;
    private Vector3[] _latestPoints;
    private object _lock = new object();

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _subSocket = new SubscriberSocket();
        _subSocket.Connect($"tcp://127.0.0.1:{connectToPort}");
        _subSocket.Subscribe(topic);

        _running = true;
        _listenerThread = new Thread(ReceiveLoop);
        _listenerThread.IsBackground = true;
        _listenerThread.Start();
    }

    private void ReceiveLoop()
    {
        while (_running)
        {
            try
            {
                NetMQMessage message = _subSocket.ReceiveMultipartMessage();

                if (message.FrameCount < 3)
                {
                    Debug.LogWarning("Malformed pointcloud message");
                    continue;
                }

                byte[] countBytes = message[1].ToByteArray();
                int pointCount = BitConverter.ToInt32(countBytes, 0);

                byte[] dataBytes = message[2].ToByteArray();

                if (pointCount > 0)
                {
                    Vector3[] newPoints = new Vector3[pointCount];

                    for (int i = 0; i < pointCount; i++)
                    {
                        int offset = i * 12; 
                        float x = BitConverter.ToSingle(dataBytes, offset);
                        float y = BitConverter.ToSingle(dataBytes, offset + 4);
                        float z = BitConverter.ToSingle(dataBytes, offset + 8);

                        newPoints[i] = new Vector3(x, y, z);
                    }

                    lock (_lock)
                    {
                        _latestPoints = newPoints;
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"ZeroMQ Error: {ex.Message}");
            }
        }
    }
    
    void Update()
    {
        lock (_lock)
        {
            if (_latestPoints != null)
            {
                OnPointcloudReceived?.Invoke(_latestPoints);
                _latestPoints = null;
            }
        }
    }

    void OnDestroy()
    {
        _running = false;
        _listenerThread?.Join();
        _subSocket?.Close();
        NetMQConfig.Cleanup();
    }
}