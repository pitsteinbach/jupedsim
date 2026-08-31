//! K=4 parallel Fast Sweeping Method.
//!
//! Processes four eikonal problems simultaneously, reading the speed field once per cell
//! per sweep instead of once per destination. The interleaved layout `u[cell * 4 + k]`
//! keeps all four destination values for a cell contiguous, which lets LLVM emit f64x4
//! loads/stores (NEON on aarch64, AVX2 on x86_64) for the inner update loop.

/// Source index value meaning "no destination in this lane".
/// The lane is initialised to INFINITY and stays that way.
pub const ABSENT: u32 = u32::MAX;

/// Solve 4 eikonal problems simultaneously, reading `speed_field` once per cell per sweep.
///
/// # Layout
/// `group_out` has length `4 * n` where `n = width * height`. On return it contains
/// four contiguous travel-time grids:
/// ```text
/// [ dest 0: n doubles | dest 1: n doubles | dest 2: n doubles | dest 3: n doubles ]
/// ```
/// i.e. `group_out[k * n + cell]` = travel time for destination `k` at `cell`.
///
/// `sources` must have exactly 4 entries. Use [`ABSENT`] to mark unused lanes.
pub fn solve_k4_into(
    group_out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32; 4],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let n = width * height;
    debug_assert_eq!(group_out.len(), 4 * n);
    debug_assert_eq!(speed_field.len(), n);

    // Interleaved working buffer: u_il[idx * 4 + k] = travel time for destination k at cell idx.
    // Interleaving keeps the 4 values for one cell contiguous — cache-friendly for f64x4 loads.
    let mut u_il = vec![f64::INFINITY; 4 * n];

    for k in 0..4usize {
        if sources[k] != ABSENT {
            u_il[sources[k] as usize * 4 + k] = 0.0;
        }
    }

    // Two passes × four sweep directions = same convergence criterion as scalar FSM.
    for _ in 0..2 {
        sweep(&mut u_il, speed_field, width, height, cell_size, false, false);
        sweep(&mut u_il, speed_field, width, height, cell_size, false, true);
        sweep(&mut u_il, speed_field, width, height, cell_size, true, false);
        sweep(&mut u_il, speed_field, width, height, cell_size, true, true);
    }

    // Transpose interleaved [n][4] → per-destination [4][n] for the caller.
    // Written as a loop over cells so the inner `k in 0..4` auto-vectorises.
    for cell in 0..n {
        for k in 0..4 {
            group_out[k * n + cell] = u_il[cell * 4 + k];
        }
    }
}

fn sweep(
    u: &mut [f64], // interleaved, length n * 4
    speed: &[f64],
    width: usize,
    height: usize,
    h: f64,
    i_rev: bool,
    j_rev: bool,
) {
    for i in idx_iter(height, i_rev) {
        for j in idx_iter(width, j_rev) {
            let idx = i * width + j;
            let f = speed[idx];
            if f <= 0.0 {
                continue;
            }
            let cost = h / f;
            let base = idx * 4;

            // Load x-neighbour values (4 destinations each).
            let ax = if j > 0 { load4(u, (idx - 1) * 4) } else { INF4 };
            let bx = if j + 1 < width { load4(u, (idx + 1) * 4) } else { INF4 };
            let a = min4(ax, bx);

            // Load y-neighbour values.
            let ay = if i > 0 { load4(u, (idx - width) * 4) } else { INF4 };
            let by = if i + 1 < height { load4(u, (idx + width) * 4) } else { INF4 };
            let b = min4(ay, by);

            // Four Godunov updates sharing one `cost` — LLVM vectorises this loop to f64x4.
            for k in 0..4 {
                let c = godunov(a[k], b[k], cost);
                if c < u[base + k] {
                    u[base + k] = c;
                }
            }
        }
    }
}

/// Upwind Godunov update (scalar); inlined into the k-loop so LLVM sees 4 independent
/// instances and emits f64x4 instructions.
#[inline(always)]
fn godunov(a: f64, b: f64, cost: f64) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    let u1 = lo + cost;
    let disc = 2.0 * cost * cost - (a - b) * (a - b);
    // Clamp disc before sqrt to avoid NaN; the select below prefers u1 when disc < 0.
    let u2 = (a + b + disc.max(0.0).sqrt()) * 0.5;
    // Bitwise AND keeps the branch out of the select, helping LLVM emit FSEL/VBLEND.
    let use_2d = (u1 > hi) & (disc >= 0.0);
    if use_2d { u2 } else { u1 }
}

const INF4: [f64; 4] = [f64::INFINITY; 4];

#[inline(always)]
fn load4(slice: &[f64], base: usize) -> [f64; 4] {
    [slice[base], slice[base + 1], slice[base + 2], slice[base + 3]]
}

#[inline(always)]
fn min4(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

// Iterator helpers — avoids Box<dyn Iterator> per sweep direction.
enum IdxIter {
    Fwd(std::ops::Range<usize>),
    Rev(std::iter::Rev<std::ops::Range<usize>>),
}
impl Iterator for IdxIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            IdxIter::Fwd(r) => r.next(),
            IdxIter::Rev(r) => r.next(),
        }
    }
}
fn idx_iter(len: usize, rev: bool) -> IdxIter {
    if rev {
        IdxIter::Rev((0..len).rev())
    } else {
        IdxIter::Fwd(0..len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k4_matches_scalar_fsm() {
        let w = 51usize;
        let h = 51usize;
        let cs = 1.0_f64;
        let speed = vec![1.0_f64; w * h];

        // Four different point sources placed symmetrically.
        let srcs = [
            (h / 4 * w + w / 4) as u32,
            (h / 4 * w + 3 * w / 4) as u32,
            (3 * h / 4 * w + w / 4) as u32,
            (3 * h / 4 * w + 3 * w / 4) as u32,
        ];

        let n = w * h;
        let mut group_out = vec![0.0_f64; 4 * n];
        solve_k4_into(&mut group_out, &speed, &srcs, w, h, cs);

        // Verify each destination's travel time at its own source is (near) zero.
        for (k, &src) in srcs.iter().enumerate() {
            let t = group_out[k * n + src as usize];
            assert!(t < 1e-9, "dest {k}: source travel time {t} should be ~0");
        }

        // Cross-check against scalar FSM for destination 0.
        let scalar = crate::fsm::solve(&speed, &srcs[..1], w, h, cs);
        for cell in 0..n {
            let k4_val = group_out[cell];
            let sc_val = scalar[cell];
            let diff = (k4_val - sc_val).abs();
            assert!(
                diff < 1e-9,
                "cell {cell}: k4={k4_val} scalar={sc_val} diff={diff}"
            );
        }
    }
}
