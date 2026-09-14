// Fill every element of u[] with +infinity using 2D dispatch.
// 2D is required because wgpu caps each dispatch dimension at 65535 workgroups;
// a 1D dispatch of ceil(k*N/256) would overflow that for large k×N.
// Linear index: idx = gid.y * nwg.x * 256 + gid.x, where nwg.x is the x
// workgroup count supplied at dispatch time via @builtin(num_workgroups).

@group(0) @binding(0) var<storage, read_write> u: array<f32>;

// bitcast is not allowed in WGSL const expressions (Naga restriction), so
// infinity is produced via a helper function called at runtime instead.
// IEEE 754 positive infinity (0x7F800000):
//   • godunov(∞, ∞, cost) = ∞  →  unreachable cells stay at ∞ forever
//   • CPU readback is a plain memcpy; f32::INFINITY as f64 == f64::INFINITY
fn gpu_inf() -> f32 { return bitcast<f32>(0x7F800000u); }

@compute @workgroup_size(256, 1, 1)
fn fill_sentinel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups)        nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 256u + gid.x;
    if idx < arrayLength(&u) {
        u[idx] = gpu_inf();
    }
}
