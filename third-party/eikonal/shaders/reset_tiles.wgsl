// Reset dirty tiles to ∞ before a warm re-solve.
//
// Each workgroup handles one tile slot from the tile_slots list (same encoding
// as active_in in fim_update.wgsl: slot = dest * num_tiles + tile_idx).
// Every thread writes gpu_inf() to its own cell.  Source cells are restored to
// 0.0 afterward by the CPU via queue.write_buffer, so this kernel does not need
// to know which cell is the source.
//
// Uniform layout: identical to fim_update.wgsl — same params_buf is reused.

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

@group(0) @binding(0) var<uniform>             params:     Params;
@group(0) @binding(1) var<storage, read_write> u:          array<f32>;
@group(0) @binding(2) var<storage, read>       tile_slots: array<u32>;
// Number of valid entries in tile_slots. The dispatch is a rounded-up 2-D grid
// (see MAX_WG_PER_DIM in fim_gpu_wgpu.rs), so the tail must be discarded.
@group(0) @binding(3) var<storage, read>       slot_count: array<u32>;

fn gpu_inf() -> f32 { return bitcast<f32>(0x7F800000u); }


@compute @workgroup_size(TILE, TILE, 1)
fn reset_dirty_tiles(
    @builtin(workgroup_id)        wg:  vec3<u32>,
    @builtin(num_workgroups)      nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Flatten the 2-D grid; discard the round-up overshoot.
    let flat_wg = wg.y * nwg.x + wg.x;
    if flat_wg >= slot_count[0] { return; }

    let tile_slot = tile_slots[flat_wg];
    let dest      = tile_slot / params.num_tiles;
    let tile_idx  = tile_slot % params.num_tiles;
    let tile_row  = tile_idx / params.num_tile_cols;
    let tile_col  = tile_idx % params.num_tile_cols;
    let global_r  = tile_row * TILE + lid.y;
    let global_c  = tile_col * TILE + lid.x;
    if global_r >= params.h || global_c >= params.w { return; }
    u[dest * params.n + global_r * params.w + global_c] = gpu_inf();
}
