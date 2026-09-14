// Seed the boundary of the region discarded by threshold_reset.
//
// After the threshold reset the field is split in two: cells below the
// destination's threshold keep their (provably still-correct) prior value, and
// cells at or above it are ∞ and must be recomputed.  FIM can only recompute them
// by propagating inward from the surviving region, so the active list has to start
// on that boundary.
//
// Seeding only the source and changed-cell tiles is not enough.  The wave then has
// to flood the whole discarded region starting from wherever the change happened,
// which (a) makes the active set explode as it spreads along the entire threshold
// contour and (b) reaches many cells by a detour rather than from the nearest
// surviving cell, leaving them too large.  Measured on the EdSheeran arena that
// showed up as worst_rel up to 1.84 -- warm nearly 3x the cold value.
//
// This pass marks every tile that owns a reset cell adjacent to a kept cell, so the
// wave starts along the whole contour at once and sweeps outward.
//
// Appends into active_in (the ping buffer) through the same tile_round tag that
// try_enqueue in fim_update.wgsl uses, so tiles already seeded by the CPU are
// deduplicated rather than added twice.

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

@group(0) @binding(0) var<uniform>             params:       Params;
@group(0) @binding(1) var<storage, read>       u:            array<f32>;
@group(0) @binding(2) var<storage, read_write> tile_round:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> active_out:   array<u32>;
@group(0) @binding(4) var<storage, read_write> active_count: atomic<u32>;


@compute @workgroup_size(256, 1, 1)
fn seed_boundary(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups)       nwg: vec3<u32>,
) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    let total = params.k * params.n;
    if i >= total {
        return;
    }
    // Only cells that were discarded can be on the inward-facing boundary.
    if u[i] < bitcast<f32>(0x7F800000u) {
        return;
    }

    let cell = i % params.n;
    let r    = cell / params.w;
    let c    = cell % params.w;
    let base = i - cell;

    // Is any 4-neighbour a surviving (finite) cell?
    var touches_kept = false;
    if c > 0u             && u[base + cell - 1u]        < bitcast<f32>(0x7F800000u) { touches_kept = true; }
    if c + 1u < params.w  && u[base + cell + 1u]        < bitcast<f32>(0x7F800000u) { touches_kept = true; }
    if r > 0u             && u[base + cell - params.w]  < bitcast<f32>(0x7F800000u) { touches_kept = true; }
    if r + 1u < params.h  && u[base + cell + params.w]  < bitcast<f32>(0x7F800000u) { touches_kept = true; }
    if !touches_kept {
        return;
    }

    // Enqueue this cell's tile for round 1, deduplicated against the CPU seeds.
    let dest = i / params.n;
    let slot = dest * params.num_tiles
             + (r / TILE) * params.num_tile_cols
             + (c / TILE);
    let old = atomicMax(&tile_round[slot], 1u);
    if old < 1u {
        let pos = atomicAdd(&active_count, 1u);
        if pos < params.active_cap {
            active_out[pos] = slot;
        }
    }
}
