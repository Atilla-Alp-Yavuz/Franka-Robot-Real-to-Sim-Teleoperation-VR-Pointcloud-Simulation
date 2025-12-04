using UnityEngine;

public class IRISVizFoveationAdapter : MonoBehaviour
{
    public FoveatedPointcloudSubscriber subscriber;
    public IrisStylePointCloudRenderer irisRenderer;

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
        if (irisRenderer != null)
            irisRenderer.UpdateCloud(pts);
    }
}