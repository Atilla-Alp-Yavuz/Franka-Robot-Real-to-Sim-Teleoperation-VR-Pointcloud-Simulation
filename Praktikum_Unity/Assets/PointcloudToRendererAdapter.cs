using UnityEngine;

public class PointcloudToRendererAdapter : MonoBehaviour
{
    public FoveatedPointcloudSubscriber subscriber;
    public SimplePointCloudRenderer rendererComponent;

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

    private void OnCloud(Vector3[] pts)
    {
        if (rendererComponent != null)
        {
            rendererComponent.UpdateCloud(pts);
        }
    }
}