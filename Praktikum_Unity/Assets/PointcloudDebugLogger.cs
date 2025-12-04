using UnityEngine;

public class PointcloudDebugLogger : MonoBehaviour
{
    public FoveatedPointcloudSubscriber subscriber;

    private int _frames;
    private int _lastCount;

    void OnEnable()
    {
        if (subscriber != null)
            subscriber.OnPointcloudReceived += OnCloud;
    }

    void OnDisable()
    {
        if (subscriber != null)
            subscriber.OnPointcloudReceived -= OnCloud;
    }

    void OnCloud(Vector3[] pts)
    {
        _frames++;
        _lastCount = pts.Length;
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 400, 20), $"Frames received: {_frames}");
        GUI.Label(new Rect(10, 30, 400, 20), $"Last cloud size: {_lastCount}");
    }
}