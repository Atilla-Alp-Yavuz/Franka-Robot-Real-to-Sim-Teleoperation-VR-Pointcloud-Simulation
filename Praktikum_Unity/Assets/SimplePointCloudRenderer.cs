using UnityEngine;

/// <summary>
/// Very simple point cloud renderer using MeshTopology.Points.
/// Call UpdateCloud() with a Vector3[] to update the cloud.
/// </summary>
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class SimplePointCloudRenderer : MonoBehaviour
{
    private Mesh _mesh;
    private Vector3[] _points = new Vector3[0];

    [Tooltip("Uniform scale applied to incoming points.")]
    public float scale = 0.1f;   // <-- new

    void Awake()
    {
        var mf = GetComponent<MeshFilter>();
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mf.mesh = _mesh;
    }

    public void UpdateCloud(Vector3[] points)
    {
        if (points == null || points.Length == 0)
        {
            _mesh.Clear();
            _points = new Vector3[0];
            return;
        }

        // apply scaling
        _points = new Vector3[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            _points[i] = points[i] * scale;
        }

        _mesh.Clear();
        _mesh.vertices = _points;

        int[] indices = new int[_points.Length];
        for (int i = 0; i < _points.Length; i++)
            indices[i] = i;

        _mesh.SetIndices(indices, MeshTopology.Points, 0, true);
        _mesh.RecalculateBounds();
    }
}