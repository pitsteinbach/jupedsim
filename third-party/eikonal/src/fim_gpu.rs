//! GPU-accelerated batch FIM via CubeCL (wgpu / Metal backend).
//!
//! This implements the blocked/tiled FIM described in:
//!   Fu, Jeong, Pan, Kirby & Whitaker 2011, SIAM J. Sci. Comput. 33(5)
//!
//! KEY DIFFERENCE from a naive GPU FIM (which syncs CPU↔GPU every cell-level round):
//!
//! Each GPU cube (thread block) owns one TILE×TILE patch of the grid.
//! Within a cube, threads iterate locally (8 passes per kernel launch)
//! using shared memory — no cross-cube communication, only sync_cube() barriers.
//!
//! Active-tile tracking:
//!   tile_round[dest * num_tiles + tile_idx] == current_round  → tile runs this launch
//!   If any cell in a tile updates, write current_round+1 to self and ≤4 neighbour tiles.
//!
//! Convergence strategy: pre-encode max_rounds = 2 × ceil(diagonal/TILE) launches
//! with NO per-round CPU readback. All rounds accumulate in one Metal command buffer.
//! One GPU→CPU sync at the very end replaces ~100 intermediate syncs.
//!
//! TILE is a compile-time constant (required by SharedMemory::new).

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use std::sync::OnceLock;

// ── Singleton GPU client ──────────────────────────────────────────────────────

type Client = ComputeClient<WgpuRuntime>;

fn gpu_client() -> &'static Client {
    static C: OnceLock<Client> = OnceLock::new();
    C.get_or_init(|| WgpuRuntime::client(&WgpuDevice::DefaultDevice))
}

// ── Algorithm constants ───────────────────────────────────────────────────────

/// Tile edge length in cells. Each GPU cube owns one TILE×TILE patch.
/// CubeDim must be TILE_SQ. SharedMemory::new sizes must match TILE_SQ exactly
/// as direct usize literals (not comptime!() wrappers — CubeCL limitation).
const TILE: usize = 16;
const TILE_SQ: usize = TILE * TILE;  // 256 threads per cube; matches SharedMemory::new(256usize)

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
    tile_round:    &mut Array<u32>,  // [K * num_tiles] round tags
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
    if tile_round[tile_slot] != round {
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
    let mut smem = SharedMemory::<f32>::new(256usize);  // TILE_SQ = 16*16

    smem[unit] = if valid { u[base + gidx] } else { SENTINEL.into() };

    // Shared flag: was any cell in this tile updated? (benign write-1 race is fine)
    let mut smem_updated = SharedMemory::<u32>::new(1usize);
    if unit == 0 { smem_updated[0usize] = 0u32; }

    sync_cube();

    // ── Local iteration: PASSES Gauss-Seidel steps within this tile ───────────
    let f       = speed[gidx];  // safe: gidx is always in [0, n)
    let is_src  = valid && gidx == src;
    let is_wall = f == 0.0_f32;

    for _pass in 0u32..16u32 {
        if valid && !is_src && !is_wall {
            // Left neighbour: smem if within tile, else global memory.
            let a_l = if local_c > 0 {
                smem[unit - 1]
            } else if global_c > 0 {
                u[base + global_r * w as usize + global_c - 1]
            } else {
                SENTINEL.into()
            };

            // Right neighbour
            let a_r = if local_c + 1 < TILE && global_c + 1 < w as usize {
                smem[unit + 1]
            } else if global_c + 1 < w as usize {
                u[base + global_r * w as usize + global_c + 1]
            } else {
                SENTINEL.into()
            };

            // Up neighbour (row - 1)
            let b_u = if local_r > 0 {
                smem[unit - TILE]
            } else if global_r > 0 {
                u[base + (global_r - 1) * w as usize + global_c]
            } else {
                SENTINEL.into()
            };

            // Down neighbour (row + 1)
            let b_d = if local_r + 1 < TILE && global_r + 1 < h as usize {
                smem[unit + TILE]
            } else if global_r + 1 < h as usize {
                u[base + (global_r + 1) * w as usize + global_c]
            } else {
                SENTINEL.into()
            };

            let a    = if a_l < a_r { a_l } else { a_r };
            let b    = if b_u < b_d { b_u } else { b_d };
            let cand = godunov_gpu(a, b, cell_size / f);
            let old  = smem[unit];
            let diff = if cand < old { old - cand } else { cand - old };

            if diff > conv_tol && cand < old {
                smem[unit] = cand;
                smem_updated[0usize] = 1u32;  // benign write race — all writers store 1
            }
        }
        sync_cube();
    }

    // ── Write tile back to global memory ─────────────────────────────────────
    if valid {
        u[base + gidx] = smem[unit];
    }
    sync_storage();

    // ── Activate neighbouring tiles for next round ────────────────────────────
    // Unit 0 alone writes to neighbour tile_round slots to keep writes tidy.
    // Multiple cubes may write the same value (round+1) to the same slot — benign.
    if smem_updated[0usize] > 0u32 {
        if unit == 0 {
            let next = round + 1u32;
            // Self: stays active
            tile_round[tile_slot] = next;

            // Up tile
            if tile_row > 0 {
                tile_round[tile_slot - num_tile_cols as usize] = next;
            }
            // Down tile
            if tile_row + 1 < num_tile_rows as usize {
                tile_round[tile_slot + num_tile_cols as usize] = next;
            }
            // Left tile
            if tile_col > 0 {
                tile_round[tile_slot - 1] = next;
            }
            // Right tile
            if tile_col + 1 < num_tile_cols as usize {
                tile_round[tile_slot + 1] = next;
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
    // Wave propagates at most TILE cells per global round (8 local passes × 1 cell/pass).
    // 2× safety factor covers obstacles and non-Manhattan paths.
    let diag_cells = ((width * width + height * height) as f64).sqrt();
    let max_rounds = ((diag_cells / TILE as f64).ceil() as u32 * 2).max(8) + 4;

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
