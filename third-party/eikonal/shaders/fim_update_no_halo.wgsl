// Tiled FIM update kernel — 16×16 shared memory, NO halo border.
//
// Baseline variant for benchmarking against fim_update.wgsl (18×18 halo).
// Border cell neighbours that fall outside this tile are read directly from
// global u[] each iteration instead of being pre-loaded into smem once.
//
// Build with: cargo build --features no_halo
//
// Everything else (workgroup size, inner iteration count, enqueue logic) is
// identical to the halo version so the only variable is the smem size and
// the per-iteration global reads for the 4×16=64 border cells.

struct Params {
    w:             u32,
    h:             u32,
    k:             u32,
    num_tile_cols: u32,
    num_tile_rows: u32,
    num_tiles:     u32,
    n:             u32,
    cell_size:     f32,
    conv_tol:      f32,
    active_cap:    u32,
    _pad0:         u32,
    _pad1:         u32,
};

@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read_write> u:           array<f32>;
@group(0) @binding(2) var<storage, read>       speed:       array<f32>;
@group(0) @binding(3) var<storage, read>       sources:     array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_round:  array<atomic<u32>>;

@group(1) @binding(0) var<storage, read>       active_in:        array<u32>;
@group(1) @binding(1) var<storage, read_write> active_out:       array<u32>;
@group(1) @binding(2) var<storage, read_write> active_out_count: atomic<u32>;

// 16×16 smem — own cells only, no halo ring.
var<workgroup> smem:          array<f32, TILE * TILE>;
var<workgroup> local_updated: atomic<u32>;

// TILE (and SMEM_W/SMEM_LEN/ITERS) come from wgsl_prelude() in
// fim_gpu_wgpu.rs. This variant keeps no halo, so its smem stride is TILE
// itself and SMEM_W is unused here.

fn gpu_inf() -> f32 { return bitcast<f32>(0x7F800000u); }

fn godunov(a: f32, b: f32, cost: f32) -> f32 {
    let lo = min(a, b);
    let hi = max(a, b);
    let u1 = lo + cost;
    if u1 <= hi {
        return u1;
    }
    let disc = 2.0 * cost * cost - (a - b) * (a - b);
    if disc >= 0.0 {
        return (a + b + sqrt(disc)) * 0.5;
    }
    return u1;
}

fn try_enqueue(slot: u32, next_round: u32, cap: u32) {
    let old = atomicMax(&tile_round[slot], next_round);
    if old < next_round {
        let pos = atomicAdd(&active_out_count, 1u);
        if pos < cap {
            active_out[pos] = slot;
        }
    }
}

@compute @workgroup_size(TILE, TILE, 1)
fn fim_update(
    @builtin(workgroup_id)           wg:   vec3<u32>,
    @builtin(local_invocation_id)    lid:  vec3<u32>,
    @builtin(local_invocation_index) unit: u32,
) {
    if unit == 0u {
        atomicStore(&local_updated, 0u);
    }
    workgroupBarrier();

    let tile_slot     = active_in[wg.x];
    let dest          = tile_slot / params.num_tiles;
    let tile_idx      = tile_slot % params.num_tiles;
    let current_round = atomicLoad(&tile_round[tile_slot]);

    let tile_row = tile_idx / params.num_tile_cols;
    let tile_col = tile_idx % params.num_tile_cols;
    let local_r  = lid.y;
    let local_c  = lid.x;
    let global_r = tile_row * TILE + local_r;
    let global_c = tile_col * TILE + local_c;
    let valid    = global_r < params.h && global_c < params.w;
    let base     = dest * params.n;

    let clamp_r = min(global_r, params.h - 1u);
    let clamp_c = min(global_c, params.w - 1u);
    let gidx    = clamp_r * params.w + clamp_c;
    let src     = sources[dest];
    let is_src  = valid && (gidx == src);
    let f       = speed[gidx];
    let is_wall = f == 0.0;
    let cost    = select(0.0, params.cell_size / f, f > 0.0);

    // Flat smem index — stride is TILE (16), no halo offset.
    let si = local_r * TILE + local_c;

    // Load own cell into smem.
    smem[si] = select(gpu_inf(), u[base + gidx], valid);

    workgroupBarrier();

    // ── 16 inner iterations ───────────────────────────────────────────────────
    // Interior neighbours come from smem (same as halo version).
    // Border neighbours that fall outside this tile are read from global u[]
    // every iteration — these reads return pre-round values since writes happen
    // only after all iterations, matching the halo version's behaviour.
    for (var iter: u32 = 0u; iter < 16u; iter++) {
        if valid && !is_src && !is_wall {
            // Left neighbour
            var nl: f32;
            if local_c == 0u {
                nl = select(gpu_inf(),
                            select(gpu_inf(), u[base + global_r * params.w + (global_c - 1u)], global_c > 0u),
                            valid);
            } else {
                nl = smem[si - 1u];
            }

            // Right neighbour
            var nr: f32;
            if local_c == TILE - 1u {
                let rc = global_c + 1u;
                nr = select(gpu_inf(),
                            select(gpu_inf(), u[base + global_r * params.w + rc], rc < params.w),
                            valid);
            } else {
                nr = smem[si + 1u];
            }

            // Top neighbour
            var nt: f32;
            if local_r == 0u {
                nt = select(gpu_inf(),
                            select(gpu_inf(), u[base + (global_r - 1u) * params.w + global_c], global_r > 0u),
                            valid);
            } else {
                nt = smem[si - TILE];
            }

            // Bottom neighbour
            var nb: f32;
            if local_r == TILE - 1u {
                let br = global_r + 1u;
                nb = select(gpu_inf(),
                            select(gpu_inf(), u[base + br * params.w + global_c], br < params.h),
                            valid);
            } else {
                nb = smem[si + TILE];
            }

            let a    = min(nl, nr);
            let b    = min(nt, nb);
            let cand = godunov(a, b, cost);
            if cand < smem[si] - params.conv_tol {
                smem[si] = cand;
                atomicStore(&local_updated, 1u);
            }
        }
        workgroupBarrier();
    }

    if valid {
        u[base + gidx] = smem[si];
    }
    storageBarrier();

    if unit == 0u {
        let was_updated = atomicLoad(&local_updated);
        if was_updated != 0u {
            let next_round = current_round + 1u;
            let cap        = params.active_cap;
            try_enqueue(tile_slot, next_round, cap);
            if tile_row > 0u {
                try_enqueue(dest * params.num_tiles + (tile_row - 1u) * params.num_tile_cols + tile_col, next_round, cap);
            }
            if tile_row + 1u < params.num_tile_rows {
                try_enqueue(dest * params.num_tiles + (tile_row + 1u) * params.num_tile_cols + tile_col, next_round, cap);
            }
            if tile_col > 0u {
                try_enqueue(dest * params.num_tiles + tile_row * params.num_tile_cols + (tile_col - 1u), next_round, cap);
            }
            if tile_col + 1u < params.num_tile_cols {
                try_enqueue(dest * params.num_tiles + tile_row * params.num_tile_cols + (tile_col + 1u), next_round, cap);
            }
        }
    }
}
