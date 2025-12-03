using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

// Add this to your "AR Camera" or "XR Origin" GameObject
public class IrisGazeSender : MonoBehaviour
{
    private PublisherSocket _pubSocket;
    public string topic = "gaze";
    public string port = "5556"; // Port for sending OUT to Python

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _pubSocket = new PublisherSocket();
        _pubSocket.Bind($"tcp://*:{port}");
    }

    void Update()
    {
        if (_pubSocket != null)
        {
            // Get Headset Transform
            Vector3 camPos = transform.position;
            Vector3 camDir = transform.forward;

            // Protocol: "pos_x,pos_y,pos_z,dir_x,dir_y,dir_z"
            // Simple string CSV is fast enough for 6 floats
            string message = $"{camPos.x},{camPos.y},{camPos.z},{camDir.x},{camDir.y},{camDir.z}";
            
            _pubSocket.SendMoreFrame(topic).SendFrame(message);
        }
    }

    void OnDestroy()
    {
        _pubSocket?.Close();
        NetMQConfig.Cleanup();
    }
}