using UnityEngine;

[RequireComponent(typeof(ParticleSystem))]
public class IrisStylePointCloudRenderer : MonoBehaviour
{
    private ParticleSystem _particleSystem;
    private ParticleSystem.Particle[] _particles;

    [Header("Point Appearance")]
    public float defaultSize = 0.1f;          // BIG to start with
    public Color defaultColor = Color.cyan;   // Bright

    [Tooltip("Uniform scale applied to incoming points.")]
    public float scale = 0.1f;                // you can tweak later

    void Awake()
    {
        _particleSystem = GetComponent<ParticleSystem>();

        // Force sensible settings
        var main = _particleSystem.main;
        main.loop = false;
        main.playOnAwake = false;
        main.startSpeed = 0f;
        main.startLifetime = Mathf.Infinity;      // never die
        main.maxParticles = 100000;              // allow large clouds

        var emission = _particleSystem.emission;
        emission.enabled = false;                // we control particles manually

        var shape = _particleSystem.shape;
        shape.enabled = false;

        _particleSystem.Clear();
        _particleSystem.Play();                  // ensure renderer is active
    }

    public void UpdateCloud(Vector3[] points)
    {
        if (points == null || points.Length == 0)
        {
            _particleSystem.Clear();
            return;
        }

        int count = points.Length;

        if (_particles == null || _particles.Length != count)
        {
            _particles = new ParticleSystem.Particle[count];
        }

        for (int i = 0; i < count; i++)
        {
            Vector3 p = points[i] * scale;

            _particles[i].position   = p;
            _particles[i].startColor = defaultColor;
            _particles[i].startSize  = defaultSize;
        }

        _particleSystem.SetParticles(_particles, count);

        if (!_particleSystem.isPlaying)
            _particleSystem.Play();
    }
}