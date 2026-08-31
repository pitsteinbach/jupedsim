// Fill every element of u[] with SENTINEL (1e30) using 2D dispatch.
// 2D is required because wgpu caps each dispatch dimension at 65535 workgroups;
// a 1D dispatch of ceil(k*N/256) would overflow that for large k×N.
// Linear index: idx = gid.y * nwg.x * 256 + gid.x, where nwg.x is the x
// workgroup count supplied at dispatch time via @builtin(num_workgroups).

@group(0) @binding(0) var<storage, read_write> u: array<f32>;

const SENTINEL: f32 = 1e30;

@compute @workgroup_size(256, 1, 1)
fn fill_sentinel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups)        nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 256u + gid.x;
    if idx < arrayLength(&u) {
        u[idx] = SENTINEL;
    }
}
