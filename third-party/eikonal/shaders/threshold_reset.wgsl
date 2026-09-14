// Threshold reset pass for warm FIM restart.
//
// Theory: in the eikonal solution every cell on the optimal path from source S
// to cell x has travel time ≤ u[x] (the path is monotone).  Therefore a cell
// whose prior value u[x] < u_threshold cannot have an increased-cost changed
// cell on its path, and is provably safe to keep.  Cells with u[x] ≥ threshold
// may use a now-slower path and are reset to ∞ so FIM recomputes them from the
// correct boundary formed by the safe inner region.
//
// The threshold is PER DESTINATION: threshold[d] = min(prior_d[c]) over changed
// cells c, using destination d's own prior field.  A single global minimum across
// all destinations is catastrophic here -- one changed cell close to any one
// destination's source drags the shared threshold to near zero and resets almost
// the entire field for *every* destination.  Measured on the EdSheeran arena with
// k=11..16, the global form left up to 99% of cells disagreeing with a cold solve
// and degraded further on each successive warm restart.
//
// Dispatch: 2-D grid (wg_x, wg_y) with wg_x = min(wgs, 65535) and wg_y = ceil(wgs / wg_x).
// stride_x = wg_x * 256 is packed into ThresholdParams so the shader reconstructs the flat
// index as i = gid.y * stride_x + gid.x, staying within the 65 535 per-dimension Metal limit.

struct ThresholdParams {
    total_elems: u32,   // k * n
    n:           u32,   // cells per destination, to recover d = i / n
    stride_x:    u32,   // wg_x * 256 — needed to reconstruct flat index from 2D dispatch
    _pad:        u32,   // 16-byte uniform alignment
};

@group(0) @binding(0) var<uniform>             tp:         ThresholdParams;
@group(0) @binding(1) var<storage, read_write> u:          array<f32>;
@group(0) @binding(2) var<storage, read>       thresholds: array<f32>;

fn gpu_inf() -> f32 { return bitcast<f32>(0x7F800000u); }

@compute @workgroup_size(256, 1, 1)
fn threshold_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Flat index reconstructed from 2D dispatch: i = row * stride_x + col_in_row.
    let i = gid.y * tp.stride_x + gid.x;
    if i >= tp.total_elems { return; }
    let t = thresholds[i / tp.n];
    if u[i] >= t {
        u[i] = gpu_inf();
    }
}
