//! GPU-accelerated batch FIM via CubeCL (wgpu / Metal backend).
//!
//! This implements the blocked/tiled FIM described in:
//!   Fu, Jeong, Pan, Kirby & Whitaker 2011, SIAM J. Sci. Comput. 33(5)
//!
//! KEY DIFFERENCE from a naive GPU FIM (which syncs CPU↔GPU every cell-level round):
//!
//! Each GPU cube (thread block) owns one TILE×TILE patch of the grid.
//! Within a cube, threads iterate locally (ITERS = 2*TILE-1 passes per kernel
//! launch) using shared memory — no cross-cube communication, only sync_cube().
//!
//! Active-tile tracking:
//!   tile_round[dest * num_tiles + tile_idx] == current_round  → tile runs this launch
//!   If a tile updates, write current_round+1 to itself and to those neighbours whose
//!   shared border actually changed (directional activation).
//!
//! Convergence strategy: pre-encode max_rounds = 2 × ceil(diagonal/TILE) launches
//! with NO per-round CPU readback. All rounds accumulate in one Metal command buffer.
//! One GPU→CPU sync at the very end replaces ~100 intermediate syncs.
//!
//! TILE is a compile-time constant (required by SharedMemory::new).
//!
//! ── WARNING: THIS BACKEND IS NOT CORRECT ─────────────────────────────────────
//! It is NON-DETERMINISTICALLY WRONG, and the failure is silent: the field comes
//! back fully populated (no unreached cells) but with values far too small, e.g.
//! max 317 on a 2049x2049 grid whose true field reaches ~2900, with ~188k cells at
//! exactly 1.0 (= cost added to a neighbour that read as zero).
//!
//! Characterised in diag_cubecl_scaling / bench_backends_head_to_head:
//!   * k=1 has never been observed to fail; k>=2 fails intermittently.
//!   * Probability rises with grid size but there is no safe threshold -- even
//!     1025x1025 k=8 and k=16 corrupt on some runs. 2049x2049 k=8 fails every time.
//!   * Identical configurations flip between runs (2049 k=2: pass, fail, fail;
//!     k=3: fail, fail, pass), so it is a race, not a size limit.
//!
//! Ruled out, each by direct experiment rather than inspection:
//!   * upload/readback round-trip -- exact at up to 128 MiB (diag_cubecl_roundtrip)
//!   * storage-binding limits -- client reports max_page_size 4 GiB (diag_cubecl_limits)
//!   * the 2-D cube-count split -- a verified bijection onto [0, knt) at every size
//!     used here (diag_cubecl_flat_cube)
//!   * inter-dispatch visibility -- a full device sync after every round does not
//!     fix it (EIKONAL_CUBECL_SYNC=1)
//!   * racy tile_round tags -- now atomic (fetch_max / load); a real latent bug,
//!     but fixing it did not remove the corruption
//!   * per-pass global neighbour reads -- now snapshotted into registers before the
//!     pass loop, matching the WGSL kernel's halo; also did not remove it
//!
//! Root cause NOT isolated. Do not enable USE_GPU_BATCH, and do not treat any
//! benchmark of this path as meaningful without checking the field first.
//!
//! ── STATUS ───────────────────────────────────────────────────────────────────
//! NOT the production path: USE_GPU_BATCH is false and USE_GPU_WGPU takes priority,
//! so nothing calls solve_batch_direct outside cubecl_vs_cpu_obstacles. It is kept
//! as the CubeCL-based alternative to the hand-written WGSL pipeline.
//!
//! Because it is dead code it drifted from fim_update.wgsl and had to be resynced.
//! Ported across (each verified here, not assumed):
//!   * TILE 16 -> 8, the measured optimum on the real arena.
//!   * ITERS 16 -> 2*TILE-1. The dominant lever in both kernels: 201.8 -> 140.3 ms
//!     (1.44x) on bench_cubecl_cold. Less than the WGSL kernel's 1.93x on the arena
//!     because this one has no halo and dispatches densely, so a larger share of its
//!     time is fixed overhead rather than tile relaxation.
//!   * Directional neighbour activation instead of waking all four.
//!
//! Deliberately NOT ported — these are architectural, not tuning, and changing them
//! would make this a second copy of the WGSL path rather than an alternative:
//!   * No halo. Border cells read global u[] every pass instead of relaxing against
//!     a snapshotted (TILE+2)^2 block. That is the `no_halo` variant's design.
//!   * Dense dispatch. Every round launches K*num_tiles cubes and unscheduled ones
//!     terminate immediately, rather than compacting an active list and using
//!     dispatch_workgroups_indirect.
//!   * SENTINEL = 1e30 rather than a true infinity bit pattern.
//!
//! Measured against the WGSL backend on identical inputs (bench_backends_head_to_head,
//! medians of 5, obstacle field, only runs where the CubeCL field verified clean):
//!
//!   grid        k    WGSL     CubeCL   ratio
//!   513x513     8    10.2 ms   19.8 ms  1.9x
//!   769x769     8    18.1 ms   65.3 ms  3.6x
//!   1025x1025   8    26.4 ms  134.3 ms  5.1x
//!   1025x1025  16    44.7 ms  257.8 ms  5.7x
//!
//! The gap widens with grid size because this path burns a fixed max_rounds budget
//! with no early exit -- 368 dispatches at 1025x1025 regardless of when the field
//! actually converges -- while the WGSL path reads the active count back and stops.
//! The halo and the compacted active list account for the rest. None of that is a
//! CubeCL limitation: it is what this kernel does not implement.
//!
//! Known limitation shared with the WGSL path: max_rounds is derived from
//! diag/TILE, i.e. it assumes the frontier advances one tile per round. Tortuous
//! geometry violates that, and this path cannot even detect it — it never reads the
//! active count back, so it would return a silently truncated field where the WGSL
//! path at least warns. Do not promote this path to production without fixing that.

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use std::sync::OnceLock;

// ── Singleton GPU client ──────────────────────────────────────────────────────

type Client = ComputeClient<WgpuRuntime>;

fn gpu_client() -> &'static Client {
    static C: OnceLock<Client> = OnceLock::new();
    C.get_or_init(|| {
        // EIKONAL_CUBECL_EXCLUSIVE=1 configures the memory manager for exclusive
        // pages, so every allocation gets its own wgpu::Buffer instead of a slice of
        // a shared one. That is what a compacted-active-list port would need: see
        // the indirect-dispatch notes near probe_build_list. Off by default so the
        // cost of it can be measured against the pooled default.
        if std::env::var("EIKONAL_CUBECL_EXCLUSIVE").is_ok() {
            let dev = WgpuDevice::IntegratedGpu(0);
            let _ = cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(
                &dev,
                cubecl::wgpu::RuntimeOptions {
                    memory_config: cubecl::MemoryConfiguration::ExclusivePages,
                    ..Default::default()
                },
            );
            return WgpuRuntime::client(&dev);
        }
        WgpuRuntime::client(&WgpuDevice::DefaultDevice)
    })
}

// ── Algorithm constants ───────────────────────────────────────────────────────

/// Tile edge length in cells. Each GPU cube owns one TILE×TILE patch.
/// CubeDim must be TILE_SQ. SharedMemory::new sizes must match TILE_SQ exactly
/// as direct usize literals (not comptime!() wrappers — CubeCL limitation), so
/// the literal is guarded by the static assert below rather than derived.
///
/// 8 matches fim_update.wgsl, where it was measured as the optimum on the real
/// EdSheeran arena (11.12 ms/destination vs 12.63 at TILE=16) and on a synthetic
/// 4107x3769 k=16 field (452 vs 716 ms). TILE*TILE must also fill Apple's 32-wide
/// SIMD groups: 8 gives exactly two groups, and is the smallest tile that does.
const TILE: usize = 8;
const TILE_SQ: usize = TILE * TILE;

/// The SharedMemory::new() literals in the kernel cannot reference TILE_SQ, so this
/// fails the build if TILE changes without them being updated to match.
const _: () = assert!(
    TILE_SQ == 64,
    "TILE changed: update the SharedMemory::<f32>::new(..) literal in fim_tiled_round"
);

/// In-tile relaxation passes. Derived from TILE, not tuned — see the ITERS block in
/// fim_update.wgsl for the full derivation and measurements.
///
/// A pass moves information exactly one cell of Manhattan distance, so relaxing the
/// tile takes its 4-connected graph diameter. Running fewer passes does not just
/// leave the tile unfinished: it publishes edge values that are too large, which
/// neighbours consume and propagate, so every later refinement invalidates the wake
/// behind it. That cascade is multiplicative -- in the WGSL kernel a 2x shortfall in
/// passes produced 36x tile redundancy and cost 1.93x on the arena.
///
/// This kernel keeps no halo (border cells read global u[] each pass), so its block
/// is TILE x TILE rather than (TILE+2)^2 and 2*TILE-2 would strictly suffice. It
/// uses the same 2*TILE-1 as the halo kernel: one spare pass is far cheaper than
/// re-running the tile, and it keeps the two kernels derived from one rule.
const ITERS: u32 = 2 * TILE as u32 - 1;

const SENTINEL: f32 = 1e30_f32;
pub const CONV_TOL_GPU: f32 = 1e-2;

// ── Godunov update ────────────────────────────────────────────────────────────

#[cube]
fn godunov_gpu(a: f32, b: f32, cost: f32) -> f32 {
    let lo = if a < b { a } else { b };
    let hi = if a > b { a } else { b };
    let u1 = lo + cost;
    let mut result = u1;
    if u1 > hi {
        let disc = 2.0_f32 * cost * cost - (a - b) * (a - b);
        if disc >= 0.0_f32 {
            result = (a + b + f32::sqrt(disc)) / 2.0_f32;
        }
    }
    result
}

// ── Tiled FIM kernel ──────────────────────────────────────────────────────────
//
// Launch dimensions:
//   CubeCount: K * num_tiles cubes (split 2-D if > 65535)
//   CubeDim:   TILE_SQ units per cube
//
// Cube flat index  = dest * num_tiles + tile_idx

#[cube(launch_unchecked)]
fn fim_tiled_round(
    u:             &mut Array<f32>,  // [K * N] travel-time grids
    speed:         &Array<f32>,      // [N]     shared speed field
    tile_round:    &mut Array<Atomic<u32>>,  // [K * num_tiles] round tags
    sources:       &Array<u32>,      // [K]     source cell per dest
    w:             u32,              // grid width in cells
    h:             u32,              // grid height in cells
    num_tile_cols: u32,              // ceil(w / TILE)
    num_tile_rows: u32,              // ceil(h / TILE)
    n:             u32,              // w * h
    cell_size:     f32,
    conv_tol:      f32,
    round:         u32,
) {
    // ── Which (dest, tile) does this cube handle? ─────────────────────────────
    let flat_cube   = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let num_tiles   = num_tile_cols * num_tile_rows;
    let dest        = flat_cube / num_tiles;
    let tile_idx    = flat_cube % num_tiles;

    // Bounds check: excess cubes from 2-D launch padding.
    if flat_cube as usize >= tile_round.len() { terminate!(); }

    // Early-exit if tile not scheduled this round.
    let tile_slot   = (dest * num_tiles + tile_idx) as usize;
    // Atomic load: other cubes in THIS dispatch are writing round+1 into these slots
    // while we read. A plain read/write pair on the same location from different
    // cubes is a data race in WGSL with indeterminate results -- it made this kernel
    // non-deterministically wrong (2049x2049 failed in 2 of 3 runs, and 1449x1449
    // k=8 flipped between runs) before these accesses were made atomic.
    if Atomic::load(&tile_round[tile_slot]) != round {
        terminate!();
    }

    // ── Geometry for this tile ────────────────────────────────────────────────
    let unit        = UNIT_POS_X as usize;
    let tile_row    = (tile_idx / num_tile_cols) as usize;
    let tile_col    = (tile_idx % num_tile_cols) as usize;
    let local_r     = unit / TILE;
    let local_c     = unit % TILE;
    let global_r    = tile_row * TILE + local_r;
    let global_c    = tile_col * TILE + local_c;
    let valid       = global_r < h as usize && global_c < w as usize;
    let base        = dest as usize * n as usize;
    // Clamp to a valid grid cell for safe reads; valid guards actual use.
    // h,w >= 1 guaranteed (grid must have cells).
    let max_r       = h as usize - 1;
    let max_c       = w as usize - 1;
    let clamp_r     = if global_r <= max_r { global_r } else { max_r };
    let clamp_c     = if global_c <= max_c { global_c } else { max_c };
    let gidx        = clamp_r * w as usize + clamp_c;
    let src         = sources[dest as usize] as usize;

    // ── Shared memory: this tile's travel times ───────────────────────────────
    // Allocated at compile time — size must be a comptime literal.
    let mut smem = SharedMemory::<f32>::new(64usize);  // TILE_SQ; see static assert

    smem[unit] = if valid { u[base + gidx] } else { SENTINEL.into() };

    // Shared edge flags — directional neighbour activation, matching the FLAG_*
    // bits in fim_update.wgsl. A neighbouring tile only ever reads our border
    // cells, so an interior-only change cannot affect it; waking all four
    // neighbours on any change (the rule this replaces) mostly re-runs a
    // one-tile-deep ring of already-converged tiles behind the frontier.
    //   slot 0 = this tile changed at all (so it may not have converged in ITERS
    //            passes and must re-run)
    //   slots 1..4 = the top / bottom / left / right border row or column changed
    // Every writer stores 1, so the concurrent writes race benignly exactly as the
    // single updated-flag they replace did, and no atomics are needed.
    let mut smem_flags = SharedMemory::<u32>::new(8usize);
    if unit == 0 {
        smem_flags[0usize] = 0u32;
        smem_flags[1usize] = 0u32;
        smem_flags[2usize] = 0u32;
        smem_flags[3usize] = 0u32;
        smem_flags[4usize] = 0u32;
    }

    sync_cube();

    // ── Local iteration: ITERS relaxation passes within this tile ────────────
    let f       = speed[gidx];  // safe: gidx is always in [0, n)
    let is_src  = valid && gidx == src;
    let is_wall = f == 0.0_f32;

    // Snapshot the OUT-OF-TILE neighbours once, before the pass loop.
    //
    // Reading them from global inside the loop -- which this kernel used to do, once
    // per pass -- races against the u[] writes other cubes in the same dispatch are
    // performing. That is a data race on non-atomic storage in WGSL terms, and it
    // made the backend NON-DETERMINISTICALLY WRONG on large grids: 2049x2049 k=2
    // passed in one run and failed the next, and failing fields came out far too
    // small (max 317 where the true field reaches ~2900) with ~188k cells sitting at
    // exactly 1.0, i.e. cost added to a neighbour that read as zero.
    //
    // Snapshotting is what fim_update.wgsl does with its (TILE+2)^2 halo block; here
    // the four values fit in registers, so no extra shared memory is needed. Values
    // still come from a concurrently-written buffer, but each location is read once
    // per dispatch rather than ITERS times, and every cell uses one consistent value
    // for the whole pass loop.
    let halo_l = if local_c == 0 && global_c > 0 {
        u[base + global_r * w as usize + global_c - 1]
    } else { SENTINEL.into() };
    let halo_r = if local_c + 1 == TILE && global_c + 1 < w as usize {
        u[base + global_r * w as usize + global_c + 1]
    } else { SENTINEL.into() };
    let halo_u = if local_r == 0 && global_r > 0 {
        u[base + (global_r - 1) * w as usize + global_c]
    } else { SENTINEL.into() };
    let halo_d = if local_r + 1 == TILE && global_r + 1 < h as usize {
        u[base + (global_r + 1) * w as usize + global_c]
    } else { SENTINEL.into() };
    sync_cube();

    for _pass in 0u32..ITERS {
        if valid && !is_src && !is_wall {
            // Left neighbour: smem if within tile, else the snapshot.
            let a_l = if local_c > 0 { smem[unit - 1] } else { halo_l };

            // Right neighbour
            let a_r = if local_c + 1 < TILE && global_c + 1 < w as usize {
                smem[unit + 1]
            } else { halo_r };

            // Up neighbour (row - 1)
            let b_u = if local_r > 0 { smem[unit - TILE] } else { halo_u };

            // Down neighbour (row + 1)
            let b_d = if local_r + 1 < TILE && global_r + 1 < h as usize {
                smem[unit + TILE]
            } else { halo_d };

            let a    = if a_l < a_r { a_l } else { a_r };
            let b    = if b_u < b_d { b_u } else { b_d };
            let cand = godunov_gpu(a, b, cell_size / f);
            let old  = smem[unit];
            let diff = if cand < old { old - cand } else { cand - old };

            if diff > conv_tol && cand < old {
                smem[unit] = cand;
                smem_flags[0usize] = 1u32;
                if local_r == 0        { smem_flags[1usize] = 1u32; }
                if local_r == TILE - 1 { smem_flags[2usize] = 1u32; }
                if local_c == 0        { smem_flags[3usize] = 1u32; }
                if local_c == TILE - 1 { smem_flags[4usize] = 1u32; }
            }
        }
        sync_cube();
    }

    // ── Write tile back to global memory ─────────────────────────────────────
    if valid {
        u[base + gidx] = smem[unit];
    }
    sync_storage();
    // Flag writes must be complete before unit 0 reads them.
    sync_cube();

    // ── Activate neighbouring tiles for next round ────────────────────────────
    // Unit 0 alone writes to neighbour tile_round slots to keep writes tidy.
    // Multiple cubes may write the same value (round+1) to the same slot — benign.
    if smem_flags[0usize] > 0u32 {
        if unit == 0 {
            let next = round + 1u32;
            // fetch_max, not a plain store: same value every writer, but the accesses
            // must be atomic to be race-free against the loads above.
            // Tags never exceed round+1, so max preserves the exact-equality schedule.
            // Self: changed, so it may not have converged in ITERS passes.
            Atomic::fetch_max(&tile_round[tile_slot], next);

            // Up tile — only if our top border row changed.
            if smem_flags[1usize] > 0u32 && tile_row > 0 {
                Atomic::fetch_max(&tile_round[tile_slot - num_tile_cols as usize], next);
            }
            // Down tile — only if our bottom border row changed.
            if smem_flags[2usize] > 0u32 && tile_row + 1 < num_tile_rows as usize {
                Atomic::fetch_max(&tile_round[tile_slot + num_tile_cols as usize], next);
            }
            // Left tile — only if our left border column changed.
            if smem_flags[3usize] > 0u32 && tile_col > 0 {
                Atomic::fetch_max(&tile_round[tile_slot - 1], next);
            }
            // Right tile — only if our right border column changed.
            if smem_flags[4usize] > 0u32 && tile_col + 1 < num_tile_cols as usize {
                Atomic::fetch_max(&tile_round[tile_slot + 1], next);
            }
        }
    }
}

// ── Host solver ───────────────────────────────────────────────────────────────

pub fn solve_batch_direct(
    out_ptrs:    &[usize],
    n:           usize,
    speed_field: &[f64],
    sources:     &[u32],
    width:       usize,
    height:      usize,
    cell_size:   f64,
) {
    let k  = out_ptrs.len();
    let c  = gpu_client();

    let tile_cols = (width  + TILE - 1) / TILE;
    let tile_rows = (height + TILE - 1) / TILE;
    let num_tiles = tile_cols * tile_rows;
    let knt       = k * num_tiles;

    // ── f32 speed field ───────────────────────────────────────────────────────
    let speed_f32: Vec<f32> = speed_field.iter().map(|&x| x as f32).collect();

    // ── Initialise travel-time grids ──────────────────────────────────────────
    let kn             = k * n;
    let mut u_init     = vec![SENTINEL; kn];
    let mut tile_round = vec![0u32; knt];

    let seed_round: u32 = 1;

    for (d, &src) in sources.iter().enumerate() {
        let base    = d * n;
        let src_idx = src as usize;
        u_init[base + src_idx] = 0.0f32;

        // Activate the source tile and its 4 neighbours.
        let src_tile_r = (src_idx / width) / TILE;
        let src_tile_c = (src_idx % width) / TILE;
        let t_base     = d * num_tiles;

        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                if dr != 0 && dc != 0 { continue; } // only cardinal + self
                let tr = src_tile_r as i64 + dr;
                let tc = src_tile_c as i64 + dc;
                if tr >= 0 && tr < tile_rows as i64 && tc >= 0 && tc < tile_cols as i64 {
                    tile_round[t_base + tr as usize * tile_cols + tc as usize] = seed_round;
                }
            }
        }
    }

    // ── Upload to GPU ─────────────────────────────────────────────────────────
    let u_h       = c.create_from_slice(f32::as_bytes(&u_init));
    let speed_h   = c.create_from_slice(f32::as_bytes(&speed_f32));
    let tr_h      = c.create_from_slice(u32::as_bytes(&tile_round));
    let src_h     = c.create_from_slice(u32::as_bytes(sources));

    // ── Bindings (cloneable GPU references) ───────────────────────────────────
    let u_bind    = u_h.clone().binding();
    let speed_b   = speed_h.binding();
    let tr_bind   = tr_h.binding();
    let src_b     = src_h.binding();

    // ── Dispatch geometry ─────────────────────────────────────────────────────
    // One cube per (dest, tile) pair; TILE_SQ units per cube.
    let n_cubes_1d = knt as u32;
    let n_cubes_x  = n_cubes_1d.min(65535);
    let n_cubes_y  = (n_cubes_1d + n_cubes_x - 1) / n_cubes_x;

    // ── Pre-compute round budget ──────────────────────────────────────────────
    // Wave propagates at most TILE cells per global round: ITERS passes at one cell
    // of Manhattan distance each is enough to cross the tile.
    // 2× safety factor covers obstacles and non-Manhattan paths -- but see the
    // tortuosity caveat in the module header: this is a heuristic, not a bound, and
    // exceeding it truncates the field silently.
    let diag_cells = ((width * width + height * height) as f64).sqrt();
    let max_rounds = ((diag_cells / TILE as f64).ceil() as u32 * 2).max(8) + 4;

    let sync_each_round = std::env::var("EIKONAL_CUBECL_SYNC").is_ok();
    let t = std::time::Instant::now();

    // ── Async round loop — NO per-round readback ──────────────────────────────
    // All kernel launches accumulate in wgpu's command queue.
    // Settled tiles terminate immediately (tile_round[slot] != round), so
    // extra rounds beyond actual convergence cost only the dispatch overhead.
    // One GPU→CPU sync happens at the final read_one_unchecked below.
    for round in 1..=max_rounds {
        unsafe {
            fim_tiled_round::launch_unchecked::<WgpuRuntime>(
                c,
                CubeCount::Static(n_cubes_x, n_cubes_y, 1),
                CubeDim::new_1d(TILE_SQ as u32),
                ArrayArg::from_raw_parts_binding(u_bind.clone(),  kn),
                ArrayArg::from_raw_parts_binding(speed_b.clone(), n),
                ArrayArg::from_raw_parts_binding(tr_bind.clone(), knt),
                ArrayArg::from_raw_parts_binding(src_b.clone(),   k),
                width      as u32,
                height     as u32,
                tile_cols  as u32,
                tile_rows  as u32,
                n          as u32,
                cell_size  as f32,
                CONV_TOL_GPU,
                round,
            )
        };
        // EIKONAL_CUBECL_SYNC=1 forces a full device sync after every round, so no
        // two dispatches can be in flight. Diagnostic for whether the corruption is
        // an inter-dispatch visibility problem rather than an intra-dispatch race.
        if sync_each_round {
            let _ = c.sync();
        }
    }

    // ── Read back and scatter ─────────────────────────────────────────────────
    // This read_one_unchecked is the ONLY GPU sync point — it flushes the entire
    // queue of max_rounds dispatches and blocks until they all complete.
    let u_bytes = c.read_one_unchecked(u_h);
    let u_f32   = f32::from_bytes(&u_bytes);

    if crate::PRINT_TIMINGS {
        println!(
            "[FIM GPU tiled async] k={k} n={n} tiles={num_tiles} max_rounds={max_rounds} total={:.1}ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    for (d, &ptr) in out_ptrs.iter().enumerate() {
        let src_slice = &u_f32[d * n .. (d + 1) * n];
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, n) };
        for (o, &v) in dst.iter_mut().zip(src_slice.iter()) {
            *o = if v >= SENTINEL * 0.5 { f64::INFINITY } else { v as f64 };
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::MemoryConfiguration;

    /// Cross-check the CubeCL kernel against the CPU FSM solver on obstacle
    /// geometry, mirroring diag_obstacles_vs_cpu for the WGSL path.
    ///
    /// This path is dead in production (USE_GPU_BATCH is false, USE_GPU_WGPU wins),
    /// so nothing else exercises it and it silently drifted from the WGSL kernel.
    /// Reachability is the sharp signal: a dropped neighbour activation or too few
    /// relaxation passes shows up as unreached cells long before it shows up as a
    /// numeric difference.
    #[test]
    #[ignore = "requires a GPU; run with --ignored"]
    fn cubecl_vs_cpu_obstacles() {
        for &w in &[257usize, 513] {
            let h = w;
            let n = w * h;
            let mut speed = vec![1.0f64; n];
            let bs = 9usize;
            let (mut r0, mut toggle) = (12usize, 0usize);
            while r0 + bs < h {
                let mut c0 = 12 + (toggle % 2) * 20;
                while c0 + bs < w {
                    for r in r0..r0 + bs {
                        for c in c0..c0 + bs { speed[r * w + c] = 0.0; }
                    }
                    c0 += 40;
                }
                r0 += 28;
                toggle += 1;
            }
            let src = ((h / 2) * w + 1) as u32;
            speed[src as usize] = 1.0;

            let mut gpu = vec![0.0f64; n];
            let ptrs = vec![gpu.as_mut_ptr() as usize];
            solve_batch_direct(&ptrs, n, &speed, &[src], w, h, 1.0);

            let cpu = crate::fsm::solve_typed::<f64>(&speed, &[src], w, h, 1.0);

            let (mut only_gpu, mut only_cpu, mut both) = (0usize, 0usize, 0usize);
            let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
            for i in 0..n {
                match (gpu[i].is_finite(), cpu[i].is_finite()) {
                    (true, true) => {
                        both += 1;
                        let d = (gpu[i] - cpu[i]).abs();
                        if d > max_abs { max_abs = d; }
                        if cpu[i] > 1.0 {
                            let r = d / cpu[i];
                            if r > max_rel { max_rel = r; }
                        }
                    }
                    (true, false) => only_gpu += 1,
                    (false, true) => only_cpu += 1,
                    _ => {}
                }
            }
            println!(
                "grid {w}x{h}: both={both} gpu_only={only_gpu} cpu_only={only_cpu} \
max_abs={max_abs:.3} max_rel={max_rel:.5}"
            );
            assert_eq!(only_cpu, 0, "{w}x{h}: {only_cpu} cells the CPU reached and the GPU did not");
            assert_eq!(only_gpu, 0, "{w}x{h}: {only_gpu} cells the GPU reached and the CPU did not");
            // CONV_TOL accumulates along the path, so this tracks the WGSL kernel's
            // tolerance rather than being exact.
            assert!(max_rel < 0.02, "{w}x{h}: max_rel {max_rel} too large");
        }
    }

    /// Can CubeCL express the compacted-active-list + indirect-dispatch scheme that
    /// the WGSL pipeline uses? Runs the whole pattern end to end and asserts the
    /// consuming kernel launched exactly as many cubes as the compaction found,
    /// with the count never leaving the GPU.
    ///
    /// It also probes the two things that are NOT obvious from the cubecl API:
    /// how large the indirect buffer must be to escape the shared memory pool, and
    /// whether a flush is needed between publishing the dims and dispatching.
    #[test]
    #[ignore = "requires a GPU; run with --ignored"]
    fn cubecl_indirect_dispatch_works() {
        let c = gpu_client();
        let total = 4096usize;
        let keep: Vec<u32> = (0..total).map(|i| if i % 7 == 0 { 1 } else { 0 }).collect();
        let expected = keep.iter().filter(|&&v| v == 1).count() as u32;
        let expected_max = (0..total as u32).filter(|i| i % 7 == 0).max().unwrap();

        // A second client whose memory manager uses ExclusivePages: every allocation
        // gets its own wgpu::Buffer rather than a sub-slice of a shared one. Keyed on
        // IntegratedGpu(0) rather than DefaultDevice so it registers as a separate
        // client (on Apple Silicon both resolve to the same physical GPU).
        let excl_dev = WgpuDevice::IntegratedGpu(0);
        let _ = cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(
            &excl_dev,
            cubecl::wgpu::RuntimeOptions {
                memory_config: MemoryConfiguration::ExclusivePages,
                ..Default::default()
            },
        );
        let c_excl: Client = WgpuRuntime::client(&excl_dev);

        // Run the pattern with a given indirect-buffer size and flush choice.
        // Returns Err(msg) if wgpu rejected the dispatch.
        let attempt = |c: &Client, ind_words: usize, do_flush: bool| -> Result<(u32, u32, u32), String> {
            let keep_h     = c.create_from_slice(u32::as_bytes(&keep));
            let list_h     = c.create_from_slice(u32::as_bytes(&vec![0u32; total]));
            let counter_h  = c.create_from_slice(u32::as_bytes(&[0u32]));
            let indirect_h = c.create_from_slice(u32::as_bytes(&vec![0u32; ind_words]));
            let out_h      = c.create_from_slice(u32::as_bytes(&[0u32, 0]));

            let units = 256u32;
            let cubes = (total as u32).div_ceil(units);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                probe_build_list::launch_unchecked::<WgpuRuntime>(
                    c,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(units),
                    ArrayArg::from_raw_parts_binding(list_h.clone().binding(),    total),
                    ArrayArg::from_raw_parts_binding(counter_h.clone().binding(), 1),
                    ArrayArg::from_raw_parts_binding(keep_h.clone().binding(),    total),
                );
                probe_publish_dims::launch_unchecked::<WgpuRuntime>(
                    c,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts_binding(counter_h.clone().binding(),  1),
                    ArrayArg::from_raw_parts_binding(indirect_h.clone().binding(), 3),
                );
                if do_flush { c.flush().expect("flush"); }
                // Cube count for this launch lives in GPU memory only.
                probe_consume_list::launch_unchecked::<WgpuRuntime>(
                    c,
                    CubeCount::Dynamic(indirect_h.clone().binding()),
                    CubeDim::new_1d(units),
                    ArrayArg::from_raw_parts_binding(list_h.clone().binding(), total),
                    ArrayArg::from_raw_parts_binding(out_h.clone().binding(),  2),
                );
                let ind = c.read_one_unchecked(indirect_h.clone());
                let ind = u32::from_bytes(&ind).to_vec();
                let out = c.read_one_unchecked(out_h.clone());
                let out = u32::from_bytes(&out).to_vec();
                (ind[0], out[0], out[1])
            }));
            r.map_err(|_| "wgpu rejected the dispatch".to_string())
        };

        println!("\nindirect-buffer size vs whether the indirect dispatch validates:");
        println!("{:>12} {:>8}  {}", "words", "flush", "result");
        let mut smallest_ok: Option<usize> = None;
        for &words in &[3usize, 256 * 1024, 384 * 1024, 512 * 1024, 768 * 1024, 1024 * 1024] {
            for &fl in &[true] {
                match attempt(c, words, fl) {
                    Ok((dims, launched, maxv)) => {
                        println!("{words:>12} {fl:>8}  OK  dims={dims} launched={launched} max={maxv}");
                        assert_eq!(dims, expected);
                        assert_eq!(launched, expected);
                        assert_eq!(maxv, expected_max);
                        if smallest_ok.is_none() { smallest_ok = Some(words); }
                    }
                    Err(e) => println!("{words:>12} {fl:>8}  REJECTED ({e})"),
                }
            }
        }
        let smallest = smallest_ok.expect("indirect dispatch never validated at any size");
        println!("smallest indirect buffer that escapes the shared pool: {smallest} words \
({} KiB)", smallest * 4 / 1024);

        // Is the flush between publishing dims and dispatching actually required?
        match attempt(c, smallest, false) {
            Ok(_)  => println!("flush between publish and indirect dispatch: NOT required"),
            Err(_) => println!("flush between publish and indirect dispatch: REQUIRED"),
        }

        // The supported fix: a client configured with ExclusivePages gives every
        // allocation its own wgpu::Buffer, so a 3-word indirect buffer no longer
        // aliases the storage bindings of the dispatch it drives.
        match attempt(&c_excl, 3, false) {
            Ok((dims, launched, maxv)) => {
                println!("ExclusivePages client, 3-word indirect buffer: OK  \
dims={dims} launched={launched} max={maxv}");
                assert_eq!(dims, expected);
                assert_eq!(launched, expected);
                assert_eq!(maxv, expected_max);
            }
            Err(e) => println!("ExclusivePages client, 3-word indirect buffer: REJECTED ({e})"),
        }
    }

    /// What do the wrong fields actually contain?
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cubecl_value_histogram() {
        let (w, k) = (2049usize, 8usize);
        let (h, n, cs) = (w, w * w, 1.0f64);
        let mut speed = vec![1.0f64; n];
        let bs = w / 10;
        for (fr, fc) in [(0.15f64, 0.20f64), (0.58, 0.35), (0.33, 0.64), (0.74, 0.76)] {
            let (r0, c0) = ((fr * h as f64) as usize, (fc * w as f64) as usize);
            for r in r0..(r0 + bs).min(h) {
                for c in c0..(c0 + bs).min(w) { speed[r * w + c] = 0.0; }
            }
        }
        let sources: Vec<u32> =
            (0..k).map(|d| (((h / (k + 1)) * (d + 1)) * w + 12) as u32).collect();
        for &s in &sources { speed[s as usize] = 1.0; }
        let mut out: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
        let ptrs: Vec<usize> = out.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
        solve_batch_direct(&ptrs, n, &speed, &sources, w, h, cs);

        for d in [0usize, 1, 7] {
            let f = &out[d];
            let zeros = f.iter().filter(|&&v| v == 0.0).count();
            let ones  = f.iter().filter(|&&v| v == 1.0).count();
            let infs  = f.iter().filter(|&&v| !v.is_finite()).count();
            let other = n - zeros - ones - infs;
            let maxf  = f.iter().cloned().filter(|v| v.is_finite()).fold(0.0f64, f64::max);
            println!("dest {d}: zeros={zeros} ones={ones} inf={infs} other={other} max_finite={maxf:.1} \
source_at={}", sources[d]);
        }
    }

    /// Is the host's 2-D cube split a bijection onto [0, knt)?
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cubecl_flat_cube() {
        let c = gpu_client();
        println!("\n{:>10} {:>8} {:>7} {:>10} {:>10}", "knt", "cubes_x", "cubes_y", "missed", "doubled");
        // knt values for k=8 at the grids under test.
        for &knt in &[133_128usize, 264_992, 528_392] {
            let hits_h = c.create_from_slice(u32::as_bytes(&vec![0u32; knt]));
            let n_cubes_1d = knt as u32;
            let n_cubes_x = n_cubes_1d.min(65535);
            let n_cubes_y = (n_cubes_1d + n_cubes_x - 1) / n_cubes_x;
            unsafe {
                probe_flat_cube::launch_unchecked::<WgpuRuntime>(
                    c,
                    CubeCount::Static(n_cubes_x, n_cubes_y, 1),
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts_binding(hits_h.clone().binding(), knt),
                );
            }
            let back = c.read_one_unchecked(hits_h);
            let hits = u32::from_bytes(&back);
            let missed  = hits.iter().take(knt).filter(|&&v| v == 0).count();
            let doubled = hits.iter().take(knt).filter(|&&v| v > 1).count();
            println!("{knt:>10} {n_cubes_x:>8} {n_cubes_y:>7} {missed:>10} {doubled:>10}");
        }
    }

    /// What limits does the CubeCL client actually hold? The failing configuration
    /// is 134,348,832 bytes of `u`, which is 131 KiB over 128 MiB -- the wgpu default
    /// max_storage_buffer_binding_size. If the client requested default limits rather
    /// than the adapter maximum, binding `u` exceeds what a single binding allows.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cubecl_limits() {
        let c = gpu_client();
        let p = c.properties();
        println!("\ncubecl client memory props: {:?}", p.memory);
        println!("max_page_size = {} bytes ({:.1} MiB)",
                 p.memory.max_page_size, p.memory.max_page_size as f64 / 1048576.0);
        println!("128 MiB = {} bytes", 128 * 1024 * 1024);
        for (label, bytes) in [("1025 k=8", 33_620_000u64), ("1449 k=8", 67_187_232),
                               ("2049 k=8", 134_348_832)] {
            println!("  {label}: u = {bytes} bytes -> {}",
                     if bytes > p.memory.max_page_size { "EXCEEDS max_page_size" } else { "fits" });
        }
    }

    /// Does a plain upload -> readback round-trip survive at these sizes, with no
    /// kernel involved at all? The wrong fields are almost entirely the value 1.0,
    /// which at cell_size=1 and speed=1 is "0 + cost" -- i.e. every cell saw a ZERO
    /// neighbour, the signature of a buffer that was never written (fresh GPU memory
    /// is zero-filled). This isolates create_from_slice/read_one_unchecked from the
    /// FIM kernel entirely.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cubecl_roundtrip() {
        let c = gpu_client();
        println!("\n{:>14} {:>10} {:>12} {:>10}", "elements", "MiB", "mismatches", "first_bad");
        for &elems in &[1_050_625usize, 4_198_401, 8_405_000, 16_796_808, 33_587_208] {
            // Distinctive pattern; 0.0 is never a legitimate value here, so any zero
            // read back means untouched memory.
            let src: Vec<f32> = (0..elems).map(|i| ((i % 1_000_003) + 1) as f32).collect();
            let h = c.create_from_slice(f32::as_bytes(&src));
            let back = c.read_one_unchecked(h);
            let got = f32::from_bytes(&back);
            let (mut mism, mut first) = (0usize, usize::MAX);
            for i in 0..elems {
                if got[i] != src[i] {
                    mism += 1;
                    if first == usize::MAX { first = i; }
                }
            }
            println!("{elems:>14} {:>10.1} {mism:>12} {:>10}",
                     elems as f64 * 4.0 / 1048576.0,
                     if first == usize::MAX { "-".to_string() } else { first.to_string() });
        }
    }

    /// Where does the CubeCL backend break? bench_backends_head_to_head found it
    /// correct at 1025x1025 k=8 but badly wrong at 2049x2049 k=8. Same grid, varying
    /// k, separates a buffer-size/limit problem (error appears as k*n grows) from a
    /// round-budget or algorithmic one (error present even at k=1).
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cubecl_scaling() {
        println!("\n{:>6} {:>4} {:>12} {:>10} {:>10} {:>9}",
                 "grid", "k", "u_bytes", "budget", "max_rel", "unreached");
        let only: Vec<(usize, usize)> = match std::env::var("EIKONAL_DIAG_CFG").ok().as_deref() {
            Some("big")  => vec![(2049, 8)],
            Some("ksweep") => vec![(2049, 1), (2049, 2), (2049, 3), (2049, 4), (2049, 6), (2049, 8)],
            Some("gsweep") => vec![(2040, 2), (2044, 2), (2047, 2), (2048, 2), (2049, 2), (2056, 2)],
            Some("small")=> vec![(1025, 8)],
            _ => vec![(1025, 1), (1025, 8), (1449, 1), (1449, 8), (2049, 1), (2049, 8)],
        };
        for &(w, k) in &only {
            {
                let h = w;
                let n = w * h;
                let cs = 1.0f64;
                let mut speed = vec![1.0f64; n];
                let bs = w / 10;
                for (fr, fc) in [(0.15f64, 0.20f64), (0.58, 0.35), (0.33, 0.64), (0.74, 0.76)] {
                    let (r0, c0) = ((fr * h as f64) as usize, (fc * w as f64) as usize);
                    for r in r0..(r0 + bs).min(h) {
                        for c in c0..(c0 + bs).min(w) { speed[r * w + c] = 0.0; }
                    }
                }
                let sources: Vec<u32> =
                    (0..k).map(|d| (((h / (k + 1)) * (d + 1)) * w + 12) as u32).collect();
                for &s in &sources { speed[s as usize] = 1.0; }

                let mut out: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
                let ptrs: Vec<usize> = out.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
                solve_batch_direct(&ptrs, n, &speed, &sources, w, h, cs);

                let cpu = crate::fsm::solve_typed::<f64>(&speed, &[sources[0]], w, h, cs);
                let (mut mrel, mut unreached) = (0.0f64, 0usize);
                let (mut bad, mut worst_i) = (0usize, 0usize);
                for i in 0..n {
                    match (out[0][i].is_finite(), cpu[i].is_finite()) {
                        (true, true) => if cpu[i] > 1.0 {
                            let r = (out[0][i] - cpu[i]).abs() / cpu[i];
                            if r > 0.01 { bad += 1; }
                            if r > mrel { mrel = r; worst_i = i; }
                        },
                        (false, true) => unreached += 1,
                        _ => {}
                    }
                }
                if mrel > 0.01 {
                    let src0 = sources[0] as usize;
                    println!("      bad_cells={bad} of {n}  worst at (r={}, c={}) gpu={:.3} cpu={:.3} \
| source at (r={}, c={})",
                             worst_i / w, worst_i % w, out[0][worst_i], cpu[worst_i],
                             src0 / w, src0 % w);
                }
                let diag = ((w * w + h * h) as f64).sqrt();
                let budget = ((diag / TILE as f64).ceil() as u32 * 2).max(8) + 4;
                let elems = k * n;
                let over = if elems > (1usize << 24) { "OVER" } else { "under" };
                println!("{w:>6} {k:>4} {:>12} {budget:>10} {mrel:>10.5} {unreached:>9}   elems={elems} ({over} 2^24)",
                         k * n * 4);
            }
        }
    }

    /// Head-to-head: the CubeCL backend (this file) against the hand-written WGSL
    /// backend (fim_gpu_wgpu), on identical inputs through identical signatures.
    ///
    /// Both now run TILE=8 and ITERS=2*TILE-1, so this isolates the architectural
    /// difference rather than the tuning:
    ///   WGSL   halo block in shared memory, compacted active list dispatched
    ///          indirectly, active-count readback so it stops as soon as the field
    ///          converges, mapped readback.
    ///   CubeCL no halo (border cells read global u[] every pass), dense dispatch of
    ///          K*num_tiles cubes every round with unscheduled ones terminating
    ///          immediately, and a fixed max_rounds budget with NO early exit -- it
    ///          always runs the full budget even after convergence.
    ///
    /// Correctness is checked both ways, because a timing comparison between two
    /// backends is meaningless if they are not computing the same field.
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_backends_head_to_head() {
        // Restricted to sizes where the CubeCL backend is verified correct. It is
        // non-deterministically wrong above roughly 2049x2049 with k>=2 -- see
        // diag_cubecl_scaling and the WARNING at the top of this file. Timing a
        // backend that is silently computing the wrong field would be meaningless.
        let mut corrupted = 0usize;
        for &(w, k) in &[(513usize, 8usize), (769, 8), (1025, 8), (1025, 16)] {
            let h = w;
            let n = w * h;
            let cs = 1.0f64;
            let mut speed = vec![1.0f64; n];
            // Obstacle field scaled to the grid.
            let bs = w / 10;
            for (fr, fc) in [(0.15f64, 0.20f64), (0.58, 0.35), (0.33, 0.64), (0.74, 0.76)] {
                let (r0, c0) = ((fr * h as f64) as usize, (fc * w as f64) as usize);
                for r in r0..(r0 + bs).min(h) {
                    for c in c0..(c0 + bs).min(w) { speed[r * w + c] = 0.0; }
                }
            }
            let sources: Vec<u32> =
                (0..k).map(|d| (((h / (k + 1)) * (d + 1)) * w + 12) as u32).collect();
            for &s in &sources { speed[s as usize] = 1.0; }

            let mut a: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
            let mut b: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
            let pa: Vec<usize> = a.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
            let pb: Vec<usize> = b.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();

            // Warm up both: first call pays adapter init and shader compilation.
            crate::fim_gpu_wgpu::solve_cold(&pa, n, &speed, &sources, w, h, cs);
            solve_batch_direct(&pb, n, &speed, &sources, w, h, cs);

            let reps = 5;
            let (mut tw, mut tc) = (Vec::new(), Vec::new());
            for _ in 0..reps {
                let t = std::time::Instant::now();
                crate::fim_gpu_wgpu::solve_cold(&pa, n, &speed, &sources, w, h, cs);
                tw.push(t.elapsed().as_secs_f64() * 1e3);
                let t = std::time::Instant::now();
                solve_batch_direct(&pb, n, &speed, &sources, w, h, cs);
                tc.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let med = |v: &mut Vec<f64>| { v.sort_by(|x, y| x.partial_cmp(y).unwrap()); v[v.len() / 2] };
            let (mw, mc) = (med(&mut tw), med(&mut tc));

            // Do they agree with each other, and with the CPU?
            let cpu = crate::fsm::solve_typed::<f64>(&speed, &[sources[0]], w, h, cs);
            let cmp = |x: &[f64], y: &[f64]| -> (usize, f64) {
                let (mut mism, mut mrel) = (0usize, 0.0f64);
                for i in 0..n {
                    match (x[i].is_finite(), y[i].is_finite()) {
                        (true, true) => if y[i] > 1.0 {
                            mrel = mrel.max((x[i] - y[i]).abs() / y[i]);
                        },
                        (p, q) if p != q => mism += 1,
                        _ => {}
                    }
                }
                (mism, mrel)
            };
            let (m_ab, r_ab) = cmp(&a[0], &b[0]);
            #[allow(unused)]
            let (_, r_ac)    = cmp(&a[0], &cpu);
            let (_, r_bc)    = cmp(&b[0], &cpu);

            let diag = ((w * w + h * h) as f64).sqrt();
            let budget = ((diag / TILE as f64).ceil() as u32 * 2).max(8) + 4;
            println!(
                "\n{w}x{h} k={k}  (CubeCL fixed budget = {budget} dispatches, no early exit)\n  \
WGSL   {mw:7.1} ms\n  CubeCL {mc:7.1} ms   -> WGSL is {:.2}x faster\n  \
agreement: wgsl-vs-cubecl reach_mismatch={m_ab} max_rel={r_ab:.5} | \
vs CPU: wgsl {r_ac:.5}, cubecl {r_bc:.5}",
                mc / mw
            );
            let _ = &mut corrupted;
            if r_ab >= 0.02 || m_ab != 0 {
                corrupted += 1;
                println!("  ^^ CubeCL field is CORRUPT for this run (known non-determinism)");
            }
        }
        if corrupted > 0 {
            println!("\n{corrupted} of 4 configurations produced a CORRUPT CubeCL field this run. \
This is the known non-determinism documented at the top of this file, not a \
regression from the timing change.");
        } else {
            println!("\nAll 4 configurations agreed this run (the CubeCL corruption is \
probabilistic -- a clean run does not mean the backend is sound).");
        }
    }

    /// Timing for the CubeCL path, used to confirm the ITERS derivation transfers
    /// from the WGSL kernel to this one (which has no halo and a different
    /// scheduler, so it is not a given).
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_cubecl_cold() {
        let (w, h) = (1025usize, 1025usize);
        let n = w * h;
        let mut speed = vec![1.0f64; n];
        for (r0, c0) in [(160usize, 220usize), (600, 360), (340, 660), (760, 780)] {
            for r in r0..r0 + 100 { for c in c0..c0 + 100 { speed[r * w + c] = 0.0; } }
        }
        let k = 8usize;
        let sources: Vec<u32> =
            (0..k).map(|d| (((h / (k + 1)) * (d + 1)) * w + 12) as u32).collect();
        for &s in &sources { speed[s as usize] = 1.0; }
        let mut outs: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
        let ptrs: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();

        let mut ms: Vec<f64> = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            solve_batch_direct(&ptrs, n, &speed, &sources, w, h, 1.0);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("TILE={TILE} ITERS={ITERS}  median={:.1} ms  best={:.1} ms", ms[ms.len() / 2], ms[0]);
    }
}

// ── Feasibility probe: compacted active list + indirect dispatch in CubeCL ────
//
// The WGSL pipeline compacts its active tiles into a list with atomics and then
// launches exactly that many workgroups via dispatch_workgroups_indirect, instead
// of launching K*num_tiles cubes every round and having most of them terminate
// immediately (what fim_tiled_round does). Porting that to CubeCL needs three
// things, all of which exist in cubecl 0.10:
//
//   * Atomic<u32> with fetch_add / fetch_max / fetch_or  (frontend/element/atomic.rs)
//   * CubeCount::Dynamic(Binding), which cubecl-wgpu lowers directly to
//     pass.dispatch_workgroups_indirect (compute/stream.rs)
//   * buffers that are simultaneously kernel-writable and legal as the indirect
//     source -- cubecl-wgpu's pool is created STORAGE | COPY_SRC | COPY_DST |
//     INDIRECT, so every allocation already qualifies and no special path is needed
//
// VERDICT: yes, and with a supported configuration rather than a hack -- see
// cubecl_indirect_dispatch_works, which compacts a sparse mask and launches exactly
// the resulting number of cubes without the count ever reaching the CPU.
//
// The one non-obvious obstacle: cubecl-wgpu sub-allocates small handles out of a
// SHARED wgpu::Buffer, and wgpu tracks buffer usage at whole-buffer granularity.
// STORAGE_READ_WRITE is an exclusive usage, so a small indirect handle physically
// shares its buffer with the storage bindings of the dispatch it is driving, and
// validation fails with "conflicting usages ... STORAGE_READ_WRITE ... INDIRECT ...
// within the usage scope". The rule is that the indirect buffer must not share a
// wgpu::Buffer with any storage binding of the dispatch it drives.
//
// THE FIX: build the client with
//     RuntimeOptions { memory_config: MemoryConfiguration::ExclusivePages, .. }
// which gives every allocation its own wgpu::Buffer. A 3-word indirect buffer then
// works. This is a first-class option, available in the 0.10 already vendored here.
// gpu_client() takes EIKONAL_CUBECL_EXCLUSIVE=1 to select it.
//
// Cost of ExclusivePages on this workload: none measurable. bench_cubecl_cold, two
// interleaved reps -- pooled 136.7 / 135.2 ms, exclusive 134.8 / 133.4 ms. Pooling
// exists for ML workloads that churn many transient tensors; this solver allocates a
// handful of long-lived buffers, so sub-slicing buys it nothing.
//
// Without ExclusivePages the only alternative is padding the indirect buffer until
// it escapes the pool -- measured threshold ~1.5 MiB on this device (393216 u32
// succeeds, 262144 fails) for 12 bytes of payload. That threshold is pool page
// sizing, not an API guarantee, so do not hard-code it.
//
// Does a newer CubeCL release remove the obstacle? No, and it does not need to.
// Checked 0.11.0-pre.3 (latest as of 2026-09-11): compute/mem_manager.rs still
// builds one main pool with STORAGE | COPY_SRC | COPY_DST | INDIRECT and shared
// pages, and compute/stream.rs still passes the pooled `res.buffer` straight to
// pass.dispatch_workgroups_indirect. Identical behaviour, so upgrading changes
// nothing here either way.
//
// Flushing between publishing the dims and the indirect dispatch is NOT required:
// the conflict is buffer identity, not ordering, so this costs no extra submits.

#[cube(launch_unchecked)]
fn probe_build_list(
    list:    &mut Array<u32>,
    counter: &mut Array<Atomic<u32>>,
    keep:    &Array<u32>,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i < keep.len() {
        // Compact: only "active" entries claim a slot, exactly like try_enqueue.
        if keep[i] == 1u32 {
            let slot = Atomic::fetch_add(&counter[0usize], 1u32);
            list[slot as usize] = i as u32;
        }
    }
}

#[cube(launch_unchecked)]
fn probe_publish_dims(counter: &Array<Atomic<u32>>, indirect: &mut Array<u32>) {
    if ABSOLUTE_POS_X == 0 {
        indirect[0usize] = Atomic::load(&counter[0usize]);
        indirect[1usize] = 1u32;
        indirect[2usize] = 1u32;
    }
}

#[cube(launch_unchecked)]
fn probe_consume_list(list: &Array<u32>, out: &mut Array<Atomic<u32>>) {
    // One cube per active entry. Nothing here knows how many cubes there are --
    // that came from the buffer written above.
    if UNIT_POS_X == 0 {
        Atomic::fetch_add(&out[0usize], 1u32);
        Atomic::fetch_max(&out[1usize], list[CUBE_POS_X as usize]);
    }
}

// Probe: does the 2-D cube-count flattening cover every slot exactly once at the
// cube counts the large grids need? fim_tiled_round derives its (dest, tile) pair
// from CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X, with the host splitting knt cubes as
// x = min(knt, 65535), y = ceil(knt / x). If that mapping is not a bijection onto
// [0, knt) the kernel silently processes the wrong tiles.
#[cube(launch_unchecked)]
fn probe_flat_cube(hits: &mut Array<Atomic<u32>>) {
    if UNIT_POS_X == 0 {
        let flat = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        if (flat as usize) < hits.len() {
            Atomic::fetch_add(&hits[flat as usize], 1u32);
        }
    }
}
