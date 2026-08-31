// Tiled FIM update kernel.
//
// Each workgroup handles one active tile (16x16 cells, 256 threads).
// The active tile slot index is read from active_in[workgroup_id.x].
//
// Layout:
//   u          : f32 array [K * N], travel-time grids for all destinations
//   speed      : f32 array [N], shared speed field
//   sources    : u32 array [K], source cell index per destination
//   tile_round : atomic<u32> array [K * num_tiles], deduplication tags
//   active_in  : u32 array, tiles to process this round (slot = dest*num_tiles + tile_idx)
//   active_out : u32 array, tiles to process next round
//   active_out_count : atomic<u32>, number of tiles appended to active_out

// Uniform buffer layout: 48 bytes (3 × 16-byte rows, std140-compatible).
struct Params {
    w:             u32,
    h:             u32,
    k:             u32,
    num_tile_cols: u32,   // row 0 — 16 bytes
    num_tile_rows: u32,
    num_tiles:     u32,
    n:             u32,
    cell_size:     f32,   // row 1 — 16 bytes
    conv_tol:      f32,
    active_cap:    u32,
    _pad0:         u32,
    _pad1:         u32,   // row 2 — 16 bytes
};

@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read_write> u:           array<f32>;
@group(0) @binding(2) var<storage, read>       speed:       array<f32>;
@group(0) @binding(3) var<storage, read>       sources:     array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_round:  array<atomic<u32>>;

@group(1) @binding(0) var<storage, read>       active_in:        array<u32>;
@group(1) @binding(1) var<storage, read_write> active_out:       array<u32>;
@group(1) @binding(2) var<storage, read_write> active_out_count: atomic<u32>;

var<workgroup> smem:          array<f32, 256>;
var<workgroup> local_updated: atomic<u32>;

const TILE:     u32 = 16u;
const SENTINEL: f32 = 1e30;

// Godunov upwind differencing for the eikonal equation.
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

// Enqueue tile slot into active_out for next round, deduplicating via tile_round.
fn try_enqueue(slot: u32, next_round: u32, cap: u32) {
    let old = atomicMax(&tile_round[slot], next_round);
    if old < next_round {
        let pos = atomicAdd(&active_out_count, 1u);
        if pos < cap {
            active_out[pos] = slot;
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn fim_update(
    @builtin(workgroup_id)           wg:   vec3<u32>,
    @builtin(local_invocation_id)    lid:  vec3<u32>,
    @builtin(local_invocation_index) unit: u32,
) {
    // Initialise workgroup-level updated flag.
    if unit == 0u {
        atomicStore(&local_updated, 0u);
    }
    workgroupBarrier();

    // Decode active slot.
    let tile_slot     = active_in[wg.x];
    let dest          = tile_slot / params.num_tiles;
    let tile_idx      = tile_slot % params.num_tiles;
    let current_round = atomicLoad(&tile_round[tile_slot]);

    // Tile geometry.
    let tile_row  = tile_idx / params.num_tile_cols;
    let tile_col  = tile_idx % params.num_tile_cols;
    let local_r   = lid.y;
    let local_c   = lid.x;
    let global_r  = tile_row * TILE + local_r;
    let global_c  = tile_col * TILE + local_c;
    let valid     = global_r < params.h && global_c < params.w;
    let base      = dest * params.n;

    // Clamp out-of-bounds threads to a safe cell for uniform indexing.
    let clamp_r = min(global_r, params.h - 1u);
    let clamp_c = min(global_c, params.w - 1u);
    let gidx    = clamp_r * params.w + clamp_c;
    let src     = sources[dest];
    let is_src  = valid && (gidx == src);
    let f       = speed[gidx];
    let is_wall = f == 0.0;
    let cost    = select(0.0, params.cell_size / f, f > 0.0);

    // Load tile into shared memory.
    smem[unit] = select(SENTINEL, u[base + gidx], valid);
    workgroupBarrier();

    // 16 local Gauss-Seidel passes within this tile.
    for (var iter: u32 = 0u; iter < 16u; iter++) {
        if valid && !is_src && !is_wall {
            // Horizontal neighbours.
            var a_l: f32 = SENTINEL;
            var a_r: f32 = SENTINEL;
            if local_c > 0u {
                a_l = smem[unit - 1u];
            } else if global_c > 0u {
                a_l = u[base + global_r * params.w + (global_c - 1u)];
            }
            if (local_c + 1u < TILE) && (global_c + 1u < params.w) {
                a_r = smem[unit + 1u];
            } else if global_c + 1u < params.w {
                a_r = u[base + global_r * params.w + (global_c + 1u)];
            }

            // Vertical neighbours.
            var b_u: f32 = SENTINEL;
            var b_d: f32 = SENTINEL;
            if local_r > 0u {
                b_u = smem[unit - TILE];
            } else if global_r > 0u {
                b_u = u[base + (global_r - 1u) * params.w + global_c];
            }
            if (local_r + 1u < TILE) && (global_r + 1u < params.h) {
                b_d = smem[unit + TILE];
            } else if global_r + 1u < params.h {
                b_d = u[base + (global_r + 1u) * params.w + global_c];
            }

            let a    = min(a_l, a_r);
            let b    = min(b_u, b_d);
            let cand = godunov(a, b, cost);
            let old  = smem[unit];

            if cand < old - params.conv_tol {
                smem[unit] = cand;
                atomicStore(&local_updated, 1u);
            }
        }
        workgroupBarrier();
    }

    // Write tile back to global memory.
    if valid {
        u[base + gidx] = smem[unit];
    }
    storageBarrier();

    // Only thread 0 enqueues neighbours.
    if unit == 0u {
        let was_updated = atomicLoad(&local_updated);
        if was_updated != 0u {
            let next_round = current_round + 1u;
            let cap        = params.active_cap;

            // Self.
            try_enqueue(tile_slot, next_round, cap);

            // Up tile.
            if tile_row > 0u {
                try_enqueue(dest * params.num_tiles + (tile_row - 1u) * params.num_tile_cols + tile_col, next_round, cap);
            }
            // Down tile.
            if tile_row + 1u < params.num_tile_rows {
                try_enqueue(dest * params.num_tiles + (tile_row + 1u) * params.num_tile_cols + tile_col, next_round, cap);
            }
            // Left tile.
            if tile_col > 0u {
                try_enqueue(dest * params.num_tiles + tile_row * params.num_tile_cols + (tile_col - 1u), next_round, cap);
            }
            // Right tile.
            if tile_col + 1u < params.num_tile_cols {
                try_enqueue(dest * params.num_tiles + tile_row * params.num_tile_cols + (tile_col + 1u), next_round, cap);
            }
        }
    }
}
