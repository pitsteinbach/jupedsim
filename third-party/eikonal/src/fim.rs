use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Convergence threshold: a cell is considered settled when the Godunov
/// candidate differs from the current value by less than this amount.
const CONV_TOL: f64 = 1e-2;

/// Square tile edge length in cells.
const TILE_SIZE: usize = 32;

/// Minimum total active cells before switching from sequential to Rayon.
/// Below this count, per-round Rayon scheduling overhead exceeds the speedup.
/// Set to ~100 cells per hardware thread; independent of TILE_SIZE.
const RAYON_MIN_ACTIVE: usize = 1024;

const USE_TILED: bool = false;

/// Per-tile work state. Each tile owns a disjoint axis-aligned rectangle of
/// cells. Only the thread assigned to this tile reads/writes `active` and
/// `next_local`; no synchronisation is needed for those fields.
struct TileWork {
    tile_idx: usize,
    /// Cells to process this round.
    active: Vec<usize>,
    /// New activations discovered within this tile's bounds during the round;
    /// merged into `active` at the start of the next round.
    next_local: Vec<usize>,
}

/// Map a flat cell index to the tile that owns it.
#[inline]
fn cell_tile_idx(idx: usize, width: usize, num_tile_cols: usize) -> usize {
    (idx / width / TILE_SIZE) * num_tile_cols + (idx % width) / TILE_SIZE
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Solves the eikonal equation |∇u| = 1/f on a 2D regular grid using the
/// tile-parallel Fast Iterative Method (Jeong & Whitaker 2008).
pub fn solve(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    if USE_TILED {
        solve_tiled(speed_field, sources, width, height, cell_size)
    } else {
        let n = width * height;
        let mut u = vec![f64::INFINITY; n];
        solve_serial_into(&mut u, speed_field, sources, width, height, cell_size);
        u
    }
}

/// Allocation-free variant: initialises `out` to INFINITY and solves in place.
/// Auxiliary scratch (is_source, in_active, active-list vecs) is still allocated
/// per call, but the main output buffer comes from the caller — eliminates the
/// largest allocation contention in the parallel precomputation pattern.
pub fn solve_into(
    out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    solve_serial_into(out, speed_field, sources, width, height, cell_size);
}

fn solve_tiled(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    let n = width * height;
    let mut is_source = vec![false; n];
    for &s in sources {
        is_source[s as usize] = true;
    }

    // Build u_atomic in one pass: sources get 0, everything else INFINITY.
    let u_atomic: Vec<AtomicU64> = (0..n)
        .map(|idx| {
            AtomicU64::new(
                if is_source[idx] {
                    0.0_f64
                } else {
                    f64::INFINITY
                }
                .to_bits(),
            )
        })
        .collect();

    // Seed: walkable neighbours of every source cell.
    let mut seen = vec![false; n];
    let mut initial_active: Vec<usize> = Vec::new();
    for &s in sources {
        let idx = s as usize;
        let i = idx / width;
        let j = idx % width;
        for nb in four_neighbors(i, j, width, height) {
            if !seen[nb] && !is_source[nb] && speed_field[nb] > 0.0 {
                seen[nb] = true;
                initial_active.push(nb);
            }
        }
    }

    match run_fim_tiled(
        u_atomic,
        speed_field,
        &is_source,
        width,
        height,
        cell_size,
        initial_active,
        None,
    ) {
        Some(result) => result,
        None => {
            println!("[FIM cold] should never happen: exceeded round limit on cold solve, falling back to FSM (N={n})");
            crate::fsm::solve(speed_field, sources, width, height, cell_size)
        }
    }
}

fn solve_serial_into(
    out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let n = width * height;
    out.fill(f64::INFINITY);
    let mut is_source = vec![false; n];
    for &s in sources {
        let idx = s as usize;
        out[idx] = 0.0;
        is_source[idx] = true;
    }

    let mut in_active = vec![false; n];
    let mut current: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();

    for &s in sources {
        let idx = s as usize;
        let i = idx / width;
        let j = idx % width;
        for nb in four_neighbors(i, j, width, height) {
            if !in_active[nb] && !is_source[nb] && speed_field[nb] > 0.0 {
                in_active[nb] = true;
                current.push(nb);
            }
        }
    }

    run_fim(
        out,
        speed_field,
        &is_source,
        width,
        height,
        cell_size,
        &mut in_active,
        &mut current,
        &mut next,
    );
}

/// Re-solves using a previous travel-time field (`prior`) as initial guess.
///
/// `changed_cells` must contain the flat indices of every cell whose speed
/// changed since `prior` was computed. Only those cells and their immediate
/// neighbours are seeded into the active list, reducing active-cell work from
/// O(N) to O(K) where K is the size of the changed region.
///
/// If the correction wavefront expands far enough that the total work would
/// exceed a full cold FSM solve, falls back to FSM automatically.
pub fn solve_warm(
    speed_field: &[f64],
    prior: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    if USE_TILED {
        solve_warm_tiled(
            speed_field,
            prior,
            changed_cells,
            sources,
            width,
            height,
            cell_size,
        )
    } else {
        solve_warm_serial(
            speed_field,
            prior,
            changed_cells,
            sources,
            width,
            height,
            cell_size,
        )
    }
}

fn solve_warm_serial(
    speed_field: &[f64],
    prior: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    let n = width * height;
    let mut u = vec![f64::INFINITY; n];
    solve_warm_serial_into(&mut u, speed_field, prior, changed_cells, sources, width, height, cell_size);
    u
}

/// Allocation-free warm-start variant: copies `prior` into `out` then re-solves in place.
/// Only auxiliary scratch (is_source, in_active, active-list vecs) is allocated per call.
pub fn solve_warm_into(
    out: &mut [f64],
    speed_field: &[f64],
    prior: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    solve_warm_serial_into(out, speed_field, prior, changed_cells, sources, width, height, cell_size);
}

fn solve_warm_serial_into(
    out: &mut [f64],
    speed_field: &[f64],
    prior: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let n = width * height;
    out.copy_from_slice(prior);

    let mut is_source = vec![false; n];
    for &s in sources {
        is_source[s as usize] = true;
    }

    let mut in_active = vec![false; n];
    let mut current: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();

    for &cell in changed_cells {
        let idx = cell as usize;
        let i = idx / width;
        let j = idx % width;
        for candidate in std::iter::once(idx).chain(four_neighbors(i, j, width, height)) {
            if !in_active[candidate] && !is_source[candidate] && speed_field[candidate] > 0.0 {
                in_active[candidate] = true;
                current.push(candidate);
            }
        }
    }

    run_fim(
        out,
        speed_field,
        &is_source,
        width,
        height,
        cell_size,
        &mut in_active,
        &mut current,
        &mut next,
    );
}

fn solve_warm_tiled(
    speed_field: &[f64],
    prior: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    let t_init = Instant::now();
    let n = width * height;
    let mut is_source = vec![false; n];
    for &s in sources {
        is_source[s as usize] = true;
    }

    // Initialise u_atomic from prior in one pass; pin source cells to 0.
    let u_atomic: Vec<AtomicU64> = prior
        .iter()
        .enumerate()
        .map(|(idx, &v)| AtomicU64::new(if is_source[idx] { 0.0_f64 } else { v }.to_bits()))
        .collect();

    // Seed: changed cells + their immediate neighbours.
    let mut seen = vec![false; n];
    let mut initial_active: Vec<usize> = Vec::new();
    for &cell in changed_cells {
        let idx = cell as usize;
        let i = idx / width;
        let j = idx % width;
        for candidate in std::iter::once(idx).chain(four_neighbors(i, j, width, height)) {
            if !seen[candidate] && !is_source[candidate] && speed_field[candidate] > 0.0 {
                seen[candidate] = true;
                initial_active.push(candidate);
            }
        }
    }
    if crate::PRINT_TIMINGS {
        println!(
            "[FIM warm] N={n} K={} seed={} init={:.1}ms",
            changed_cells.len(),
            initial_active.len(),
            t_init.elapsed().as_secs_f64() * 1e3
        );
    }

    // Safety-net round limit: warm FIM is beneficial when the correction wave
    // is local. If it propagates far enough that total work exceeds ~10 FSM
    // sweeps, fall back to a fast sequential FSM cold solve.
    let max_rounds = (10.0 * (n as f64).sqrt()) as u32;

    match run_fim_tiled(
        u_atomic,
        speed_field,
        &is_source,
        width,
        height,
        cell_size,
        initial_active,
        Some(max_rounds),
    ) {
        Some(result) => result,
        None => {
            if crate::PRINT_TIMINGS {
                println!("[FIM warm] exceeded {max_rounds} rounds, falling back to FSM (N={n})");
            }
            crate::fsm::solve(speed_field, sources, width, height, cell_size)
        }
    }
}

fn run_fim(
    u: &mut [f64],
    speed_field: &[f64],
    is_source: &[bool],
    width: usize,
    height: usize,
    cell_size: f64,
    in_active: &mut [bool],
    current: &mut Vec<usize>,
    next: &mut Vec<usize>,
) {
    let t = Instant::now();
    let n = u.len();
    let seed_count = current.len();
    let mut rounds = 0u32;
    let mut total_cells_processed: usize = 0;

    while !current.is_empty() {
        rounds += 1;
        total_cells_processed += current.len();

        for &idx in current.iter() {
            in_active[idx] = false;

            if is_source[idx] || speed_field[idx] <= 0.0 {
                continue;
            }

            let i = idx / width;
            let j = idx % width;
            let a = min_neighbor_x(u, i, j, width);
            let b = min_neighbor_y(u, i, j, width, height);
            let candidate = godunov_update(a, b, cell_size / speed_field[idx]);

            if (candidate - u[idx]).abs() > CONV_TOL {
                u[idx] = candidate;
                for nb in four_neighbors(i, j, width, height) {
                    if !in_active[nb] && !is_source[nb] && speed_field[nb] > 0.0 {
                        in_active[nb] = true;
                        next.push(nb);
                    }
                }
            }
        }
        std::mem::swap(current, next);
        next.clear();
    }

    if crate::PRINT_TIMINGS {
        println!(
            "[FIM serial] rounds={rounds} seed={seed_count} N={n} cells={total_cells_processed} \
             total={:.1}ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

// ── Tile processing ───────────────────────────────────────────────────────────

/// Process one tile's active list for a single FIM round.
/// Shared by both the sequential and Rayon parallel paths.
fn process_tile(
    tile: &mut TileWork,
    u_atomic: &[AtomicU64],
    in_active: &[AtomicBool],
    speed_field: &[f64],
    is_source: &[bool],
    remote: &[Mutex<Vec<usize>>],
    width: usize,
    height: usize,
    cell_size: f64,
    num_tile_cols: usize,
) {
    tile.next_local.clear();

    for &idx in &tile.active {
        // Release ensures the deactivation is visible to any concurrent
        // AcqRel swap on other tiles (important on ARM's weaker memory model).
        in_active[idx].store(false, Ordering::Release);

        if is_source[idx] || speed_field[idx] <= 0.0 {
            continue;
        }

        let i = idx / width;
        let j = idx % width;
        let a = min_nb_x_atomic(u_atomic, i, j, width);
        let b = min_nb_y_atomic(u_atomic, i, j, width, height);
        let candidate = godunov_update(a, b, cell_size / speed_field[idx]);
        let old = f64::from_bits(u_atomic[idx].load(Ordering::Relaxed));

        if (candidate - old).abs() > CONV_TOL {
            // Safe: each cell is owned by exactly one tile, so only one
            // thread ever writes to u_atomic[idx].
            u_atomic[idx].store(candidate.to_bits(), Ordering::Relaxed);

            for nb in four_neighbors(i, j, width, height) {
                if speed_field[nb] <= 0.0 || is_source[nb] {
                    continue;
                }
                // Atomically claim nb. If it was already active, skip.
                if in_active[nb].swap(true, Ordering::AcqRel) {
                    continue;
                }
                let nb_ti = cell_tile_idx(nb, width, num_tile_cols);
                if nb_ti == tile.tile_idx {
                    tile.next_local.push(nb);
                } else {
                    remote[nb_ti].lock().unwrap().push(nb);
                }
            }
        }
    }
}

// ── Tile-parallel FIM core ────────────────────────────────────────────────────

fn run_fim_tiled(
    u_atomic: Vec<AtomicU64>,
    speed_field: &[f64],
    is_source: &[bool],
    width: usize,
    height: usize,
    cell_size: f64,
    initial_active: Vec<usize>,
    max_rounds: Option<u32>,
) -> Option<Vec<f64>> {
    let n = u_atomic.len();
    let num_tile_cols = (width + TILE_SIZE - 1) / TILE_SIZE;
    let num_tile_rows = (height + TILE_SIZE - 1) / TILE_SIZE;
    let num_tiles = num_tile_cols * num_tile_rows;

    let t_setup = Instant::now();

    // Membership flag: guards against adding the same cell to the active list twice.
    let in_active: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

    // Build per-tile work state.
    let mut tiles: Vec<TileWork> = (0..num_tiles)
        .map(|ti| TileWork {
            tile_idx: ti,
            active: Vec::new(),
            next_local: Vec::new(),
        })
        .collect();

    // Cross-tile activation queues: one per tile, written by neighbouring tiles
    // and drained sequentially between rounds.
    let remote: Vec<Mutex<Vec<usize>>> = (0..num_tiles).map(|_| Mutex::new(Vec::new())).collect();

    // Distribute initial active cells into their owning tiles.
    let initial_active_count = initial_active.len();
    for idx in initial_active {
        if !in_active[idx].swap(true, Ordering::Relaxed) {
            let ti = cell_tile_idx(idx, width, num_tile_cols);
            tiles[ti].active.push(idx);
        }
    }

    let dur_setup = t_setup.elapsed();

    // Rayon parallelism threshold: switch from sequential to parallel once the
    // active set is large enough that threads have meaningful work. For thin
    // wavefronts (early rounds or local warm restarts) sequential avoids the
    // per-round Rayon task-dispatch overhead that would otherwise dominate.
    // This is independent of TILE_SIZE so it doesn't blow up with small tiles.
    let rayon_threshold = RAYON_MIN_ACTIVE;

    let mut dur_seq = Duration::ZERO;
    let mut dur_par = Duration::ZERO;
    let mut dur_merge = Duration::ZERO;
    let mut seq_rounds = 0u32;
    let mut par_rounds = 0u32;
    let mut total_cells_processed: usize = 0;
    let t_loop = Instant::now();

    let mut rounds = 0u32;
    loop {
        rounds += 1;

        let total_active: usize = tiles.iter().map(|t| t.active.len()).sum();
        total_cells_processed += total_active;

        if total_active >= rayon_threshold {
            // ── Parallel phase ────────────────────────────────────────────────
            par_rounds += 1;
            let t = Instant::now();
            tiles.par_iter_mut().for_each(|tile| {
                process_tile(
                    tile,
                    &u_atomic,
                    &in_active,
                    speed_field,
                    is_source,
                    &remote,
                    width,
                    height,
                    cell_size,
                    num_tile_cols,
                );
            });
            dur_par += t.elapsed();
        } else {
            // ── Sequential phase ──────────────────────────────────────────────
            seq_rounds += 1;
            let t = Instant::now();
            for tile in &mut tiles {
                process_tile(
                    tile,
                    &u_atomic,
                    &in_active,
                    speed_field,
                    is_source,
                    &remote,
                    width,
                    height,
                    cell_size,
                    num_tile_cols,
                );
            }
            dur_seq += t.elapsed();
        }

        // ── Sequential merge phase ────────────────────────────────────────
        // Combine each tile's local next list with any cross-tile activations
        // it received, then check whether any tile still has work to do.
        let t = Instant::now();
        let mut any_active = false;
        for tile in &mut tiles {
            tile.active = std::mem::take(&mut tile.next_local);
            tile.active
                .extend(remote[tile.tile_idx].lock().unwrap().drain(..));
            if !tile.active.is_empty() {
                any_active = true;
            }
        }
        dur_merge += t.elapsed();

        if !any_active {
            break;
        }

        if let Some(max) = max_rounds {
            if rounds >= max {
                if crate::PRINT_TIMINGS {
                    println!(
                        "[FIM] warm exceeded round limit {max} (seed={initial_active_count} N={n})"
                    );
                }
                return None;
            }
        }
    }

    if crate::PRINT_TIMINGS {
        println!(
            "[FIM] rounds={rounds} (seq={seq_rounds}/par={par_rounds}) seed={initial_active_count} \
             N={n} cells={total_cells_processed} | \
             setup={:.1}ms loop={:.1}ms (seq={:.1}ms par={:.1}ms merge={:.1}ms)",
            dur_setup.as_secs_f64() * 1e3,
            t_loop.elapsed().as_secs_f64() * 1e3,
            dur_seq.as_secs_f64() * 1e3,
            dur_par.as_secs_f64() * 1e3,
            dur_merge.as_secs_f64() * 1e3,
        );
    }

    // Consume the atomic vec and return plain f64 values.
    Some(
        u_atomic
            .into_iter()
            .map(|a| f64::from_bits(a.into_inner()))
            .collect(),
    )
}

// ── Stencil helpers ───────────────────────────────────────────────────────────

/// Godunov upwind update: solve (u−a)₊² + (u−b)₊² = cost²
fn godunov_update(a: f64, b: f64, cost: f64) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    let u1 = lo + cost;
    if u1 <= hi {
        return u1;
    }
    let disc = 2.0 * cost * cost - (a - b) * (a - b);
    if disc >= 0.0 {
        (a + b + disc.sqrt()) / 2.0
    } else {
        u1
    }
}

fn min_nb_x_atomic(u: &[AtomicU64], i: usize, j: usize, width: usize) -> f64 {
    let left = if j > 0 {
        f64::from_bits(u[i * width + j - 1].load(Ordering::Relaxed))
    } else {
        f64::INFINITY
    };
    let right = if j + 1 < width {
        f64::from_bits(u[i * width + j + 1].load(Ordering::Relaxed))
    } else {
        f64::INFINITY
    };
    left.min(right)
}

fn min_nb_y_atomic(u: &[AtomicU64], i: usize, j: usize, width: usize, height: usize) -> f64 {
    let up = if i > 0 {
        f64::from_bits(u[(i - 1) * width + j].load(Ordering::Relaxed))
    } else {
        f64::INFINITY
    };
    let down = if i + 1 < height {
        f64::from_bits(u[(i + 1) * width + j].load(Ordering::Relaxed))
    } else {
        f64::INFINITY
    };
    up.min(down)
}

fn min_neighbor_x(u: &[f64], i: usize, j: usize, width: usize) -> f64 {
    let left = if j > 0 {
        u[i * width + j - 1]
    } else {
        f64::INFINITY
    };
    let right = if j + 1 < width {
        u[i * width + j + 1]
    } else {
        f64::INFINITY
    };
    left.min(right)
}

fn min_neighbor_y(u: &[f64], i: usize, j: usize, width: usize, height: usize) -> f64 {
    let up = if i > 0 {
        u[(i - 1) * width + j]
    } else {
        f64::INFINITY
    };
    let down = if i + 1 < height {
        u[(i + 1) * width + j]
    } else {
        f64::INFINITY
    };
    up.min(down)
}

fn four_neighbors(i: usize, j: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let mut nb = [0usize; 4];
    let mut count = 0usize;
    if j > 0 {
        nb[count] = i * width + j - 1;
        count += 1;
    }
    if j + 1 < width {
        nb[count] = i * width + j + 1;
        count += 1;
    }
    if i > 0 {
        nb[count] = (i - 1) * width + j;
        count += 1;
    }
    if i + 1 < height {
        nb[count] = (i + 1) * width + j;
        count += 1;
    }
    nb.into_iter().take(count)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_uniform_point_source() {
        let w = 101usize;
        let h = 101usize;
        let cx = w / 2;
        let cy = h / 2;
        let speed = vec![1.0_f64; w * h];
        let sources = vec![(cy * w + cx) as u32];

        let u = solve(&speed, &sources, w, h, 1.0);

        let cases = [
            ((cy, cx + 10), 0.02),
            ((cy + 10, cx), 0.02),
            ((cy + 10, cx + 10), 0.08),
        ];
        for ((i, j), tol) in cases {
            let expected = ((i as f64 - cy as f64).powi(2) + (j as f64 - cx as f64).powi(2)).sqrt();
            let got = u[i * w + j];
            assert!(
                (got - expected).abs() / expected < tol,
                "({i},{j}): expected {expected:.3}, got {got:.3} (tol {tol})"
            );
        }
    }

    #[test]
    fn cold_obstacle_isolation() {
        let w = 5usize;
        let h = 1usize;
        let mut speed = vec![1.0_f64; w * h];
        speed[2] = 0.0;
        let sources = vec![0u32];

        let u = solve(&speed, &sources, w, h, 1.0);

        assert_eq!(u[0], 0.0);
        assert!(u[1].is_finite());
        assert!(u[3].is_infinite(), "cell behind wall must be unreachable");
        assert!(u[4].is_infinite());
    }

    /// Warm start on an unchanged speed field: no cells are in changed_cells,
    /// so the active list is empty and the result equals the prior exactly.
    #[test]
    fn warm_unchanged_field() {
        let w = 51usize;
        let h = 51usize;
        let speed = vec![1.0_f64; w * h];
        let sources = vec![(25 * w + 25) as u32];

        let cold = solve(&speed, &sources, w, h, 1.0);
        // Pass empty changed_cells: nothing changed, prior is valid as-is.
        let warm = solve_warm(&speed, &cold, &[], &sources, w, h, 1.0);

        for (i, (&c, &wm)) in cold.iter().zip(warm.iter()).enumerate() {
            assert!(
                (c - wm).abs() <= CONV_TOL,
                "cell {i}: cold={c:.6}, warm={wm:.6}"
            );
        }
    }

    /// Warm start after a local speed reduction: result must match a fresh cold
    /// solve on the updated field.
    #[test]
    fn warm_local_speed_change() {
        let w = 51usize;
        let h = 51usize;
        let source_idx = (25 * w + 25) as u32;

        let speed_old = vec![1.0_f64; w * h];
        let cold_old = solve(&speed_old, &[source_idx], w, h, 1.0);

        let mut speed_new = speed_old.clone();
        for r in 38..45 {
            for c in 38..45 {
                speed_new[r * w + c] = 0.4;
            }
        }

        // Compute changed cells as the caller (C++) would.
        let changed: Vec<u32> = (0..w * h)
            .filter(|&i| (speed_new[i] - speed_old[i]).abs() > 1e-12)
            .map(|i| i as u32)
            .collect();

        let warm = solve_warm(&speed_new, &cold_old, &changed, &[source_idx], w, h, 1.0);
        let cold_new = solve(&speed_new, &[source_idx], w, h, 1.0);

        let mut max_err = 0.0_f64;
        for i in 0..w * h {
            if cold_new[i].is_finite() {
                max_err = max_err.max((warm[i] - cold_new[i]).abs());
            }
        }
        assert!(
            max_err < 5.0 * CONV_TOL,
            "warm vs cold max error {max_err:.2e} exceeds 5×CONV_TOL={:.2e}",
            5.0 * CONV_TOL
        );
    }
}
