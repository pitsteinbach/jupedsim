// Tiled FIM update kernel — TILE×TILE cells per workgroup, one thread per cell,
// in a (TILE+2)² shared-memory block with a 1-cell halo ring.  Border cells never
// read global u[] during the inner loop: all four neighbour reads are smem
// lookups, so each tile touches global memory once per round instead of once per
// iteration.
//
// Choosing TILE.  Each round runs ITERS = 2*TILE-1 iterations over all TILE² cells
// of every active tile, so work per active tile per round is ~2·TILE³.  The frontier
// advances one tile per round (rounds ≈ D/TILE) and spans L/TILE tiles, giving total
// work ≈ 2·D·L·TILE — still linear in TILE — against a dispatch count ≈ D/TILE that
// falls with it.  Which term wins depends on the frontier length L, i.e. on the
// geometry:
//
//   * Open field (no obstacles, long frontier) — work-dominated, small TILE wins.
//     Synthetic 4107x3769, k=16, uniform speed, median of 5: TILE=4 682 ms, 6 627,
//     8 452, 12 594, 16 716, 24 1575, 32 2666.
//   * Obstacle-rich pedestrian geometry (short, fragmented frontier) — the work
//     term shrinks, so the balance moves toward larger tiles. On a SMALL obstacle
//     case (1025x1025, k=1) it moves far enough that TILE=16 wins outright, 25 ms
//     vs 34 at TILE=8 — but that case is too small to fill the GPU, so it is
//     measuring round latency, not the production trade-off. Do not tune on it.
//
// TILE=8 is the default, chosen on the real scenario rather than either synthetic.
// EdSheeran arena, 2 interleaved reps, ~1380 destinations per run, per-destination
// wall time from EIKONAL_BREAKDOWN (first-call solver init excluded):
//
//   TILE            6       8      12      16
//   encode/dest 13.09    9.07   10.65   10.75   ms
//   total/dest  15.81   11.12   12.50   12.63   ms
//
// The two reps agree to ~5% (TILE=8: 11.44 / 10.79; TILE=16: 12.64 / 12.62), so the
// 12% gap to TILE=12/16 and the 30% gap to TILE=6 are both real. The floor at 8 is
// not smooth: TILE*TILE threads must fill Apple's 32-wide SIMD groups, and TILE=8
// (64 threads = exactly 2 groups) is the smallest tile that does, so it gets the
// lowest work term without wasting lanes. TILE=4 (16 threads) does the least work
// of all and is still 51% slower on the synthetic — half of every SIMD group idle.
//
// Verified not to be an artifact of round truncation: the round loop can exit with
// tiles still active (see max_rounds in fim_gpu_wgpu.rs), which would make a large
// TILE look fast while silently returning a truncated field. On this scenario at
// TILE=8 the peak is 960 of 1398 rounds and no truncation warning fires.
//
// Retune against the target geometry, not a uniform field, and not a grid too small
// to saturate the GPU.
//
// Neighbour activation is directional.  A neighbouring tile reads only our edge
// cells as its halo, so an interior-only change cannot affect it; waking all four
// neighbours on any change (the previous rule) mainly re-runs a one-tile-deep ring
// of already-converged tiles behind the frontier.  Measured: synthetic k=16 1288 vs
// 1356 ms (5%); EdSheeran arena, interleaved A/B, 2 reps each, 18.04 vs 19.90
// ms/destination (9%).  Treat the arena figure as an upper bound -- the same binary
// drifted ~8% between benchmarking sessions, so the honest range is mid-single-digit.
// Correctness gate is diag_obstacles_vs_cpu, which cross-checks the reachable set
// against the CPU FSM solver on obstacle geometry; a missed edge shows up there as
// unreached cells long before it shows up as a timing change.
//
// Precomputing cost = cell_size/speed into the speed buffer was tried and makes no
// measurable difference -- the divide happens once per thread per tile activation,
// outside the TILE-iteration loop, so it is a small slice of per-thread work.
// Measured: synthetic k=16 1353 vs 1365 ms; EdSheeran arena interleaved, the clean
// rep gave 18.30 vs 18.24 ms/destination with the GPU phase identical to three
// digits (234.4 vs 234.3 ms).  Not worth a second shader, pipeline and uniform
// buffer -- and it makes speed_buf silently hold cost, which is a trap.
//
// Dispatch overhead is NOT the lever here: adding empty compute passes to the
// round loop (2 extra Metal command buffers each) moved the EdSheeran solve by
// 0.3 ms/destination per pass — under the benchmark's 2.7 ms/destination noise
// floor.  Folding setup_indirect into this kernel would therefore buy ~1%, which
// does not justify the device-wide sync it needs.
//
// Changing TILE means editing ONE line: `const TILE` in fim_gpu_wgpu.rs. TILE,
// SMEM_W, SMEM_LEN and ITERS are injected at the top of every shader module by
// wgsl_prelude(), and this kernel's workgroup_size, its smem array length and the
// same in reset_tiles.wgsl / seed_boundary.wgsl are all written in terms of them.
// Do not reintroduce a literal tile dimension in a .wgsl file.
//
// Smem layout (row-major, stride SMEM_W = TILE+2, so 10 at TILE=8):
//   smem[0][1..TILE]        top halo row    (loaded by local_r==0 threads)
//   smem[TILE+1][1..TILE]   bottom halo row (loaded by local_r==TILE-1 threads)
//   smem[1..TILE][0]        left halo col   (loaded by local_c==0 threads)
//   smem[1..TILE][TILE+1]   right halo col  (loaded by local_c==TILE-1 threads)
//   smem[1..TILE][1..TILE]  tile's own cells (every thread loads its own cell)
//   corner cells smem[0][0] etc. are never read (Godunov is 4-connected)
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
// Number of valid entries in active_in this round, published by setup_indirect.
// The indirect dispatch grid is rounded up to a 2-D shape, so the final row of
// workgroups can overshoot; those workgroups bound-check against this value.
@group(0) @binding(5) var<storage, read>       round_count: array<u32>;

@group(1) @binding(0) var<storage, read>       active_in:        array<u32>;
@group(1) @binding(1) var<storage, read_write> active_out:       array<u32>;
@group(1) @binding(2) var<storage, read_write> active_out_count: atomic<u32>;

// (TILE+2)² smem: TILE×TILE tile cells + 1-cell halo on all four sides.
var<workgroup> smem:          array<f32, SMEM_LEN>;
var<workgroup> local_updated: atomic<u32>;
// Monotonically increasing count of cell updates, used to detect convergence.
// Monotonic on purpose: it needs no per-iteration reset, which would cost a second
// workgroupBarrier every iteration.
var<workgroup> change_count:  atomic<u32>;

// ITERS (supplied by the generated prelude, = 2*TILE-1) is NOT the same knob as
// TILE, and getting it right was the single largest win in this kernel:
//   * synthetic 4107x3769, k=16:  1288 -> 458 ms median  (2.8x)
//   * EdSheeran arena, 2 interleaved reps, ~1390 destinations per run, per
//     destination: ITERS=TILE 17.56 encode / 20.11 total ms, ITERS=2*TILE-1
//     9.09 / 11.20 -- 1.93x on the GPU phase, 1.79x on the whole precompute.
//     The arena gains less than the open field because obstacles shorten the
//     frontier, which shrinks the work term this removes.
//
// The value is the 4-connected graph diameter of the smem block, not a tuned
// constant. The halo is snapshotted at entry and never refreshed, so the loop is
// solving a fixed boundary-value problem on the interior; one Jacobi iteration
// moves influence exactly one cell of Manhattan distance. The worst case is a wave
// entering at the halo cell above the top-left interior corner and having to reach
// the bottom-right one: TILE steps down plus TILE-1 across = 2*TILE-1 = 15.
//
// At ITERS = TILE the loop covered barely half that diameter, so tiles wrote back
// half-relaxed values, re-enqueued themselves, and reloaded the whole block to
// continue -- and worse, they published too-large edge values that their neighbours
// consumed and propagated, so every later refinement invalidated the wake behind
// it. That cascade is multiplicative, which is why a 2x shortfall in iterations
// produced 36x tile redundancy rather than 2x.
//
// Measured: activations saturate exactly at 15 and are flat above it, confirming
// the tile is fully relaxed there and further iterations are pure waste.
//
//   ITERS         13       14       15       16       17
//   activations  212833   113479    56965    57152    56875
//   median ms      1268     1028      450      485      491
//
// 15 beats 16 by ~3% in interleaved reps -- one wasted iteration out of 16. Below
// 15 the cost is a cliff, not a slope. Re-derive if TILE changes.

// local_updated bits: which tiles must be re-queued after this round.
const FLAG_SELF:   u32 = 1u;   // some cell changed; tile may not have converged
const FLAG_TOP:    u32 = 2u;   // a cell in row 0 changed
const FLAG_BOTTOM: u32 = 4u;   // a cell in row TILE-1 changed
const FLAG_LEFT:   u32 = 8u;   // a cell in column 0 changed
const FLAG_RIGHT:  u32 = 16u;  // a cell in column TILE-1 changed

// bitcast is not allowed in WGSL const expressions (Naga restriction),
// so infinity is a helper function called at runtime.
fn gpu_inf() -> f32 { return bitcast<f32>(0x7F800000u); }

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

@compute @workgroup_size(TILE, TILE, 1)
fn fim_update(
    @builtin(workgroup_id)           wg:   vec3<u32>,
    @builtin(num_workgroups)         nwg:  vec3<u32>,
    @builtin(local_invocation_id)    lid:  vec3<u32>,
    @builtin(local_invocation_index) unit: u32,
) {
    if unit == 0u {
        atomicStore(&local_updated, 0u);
        atomicStore(&change_count, 0u);
    }
    workgroupBarrier();

    // Flatten the 2-D workgroup grid back into a linear active_in index.
    // Workgroups past round_count are the round-up overshoot: they must not be
    // skipped with an early `return`, because the barriers below have to stay in
    // uniform control flow.  Instead they read slot 0 (always a safe, in-bounds
    // tile) and are neutralised by folding `in_range` into `valid`, so they do no
    // stores and cannot enqueue anything.
    let flat_wg  = wg.y * nwg.x + wg.x;
    let in_range = flat_wg < round_count[0];
    let slot_idx = select(0u, flat_wg, in_range);

    // Decode active slot.
    let tile_slot     = active_in[slot_idx];
    let dest          = tile_slot / params.num_tiles;
    let tile_idx      = tile_slot % params.num_tiles;
    let current_round = atomicLoad(&tile_round[tile_slot]);

    // Tile geometry.
    let tile_row = tile_idx / params.num_tile_cols;
    let tile_col = tile_idx % params.num_tile_cols;
    let local_r  = lid.y;
    let local_c  = lid.x;
    let global_r = tile_row * TILE + local_r;
    let global_c = tile_col * TILE + local_c;
    let valid    = in_range && global_r < params.h && global_c < params.w;
    let base     = dest * params.n;

    // Clamp out-of-bounds threads to a safe cell for uniform global indexing.
    let clamp_r = min(global_r, params.h - 1u);
    let clamp_c = min(global_c, params.w - 1u);
    let gidx    = clamp_r * params.w + clamp_c;
    let src     = sources[dest];
    let is_src  = valid && (gidx == src);
    let f       = speed[gidx];
    let is_wall = f == 0.0;
    let cost    = select(0.0, params.cell_size / f, f > 0.0);

    // ── (TILE+2)² smem indices for this thread ────────────────────────────────
    // smem_r/smem_c are 1-based: smem[1..TILE][1..TILE] are the tile's own cells.
    let smem_r = local_r + 1u;
    let smem_c = local_c + 1u;
    let si     = smem_r * SMEM_W + smem_c;  // own-cell flat index in smem

    // ── Load own cell ──────────────────────────────────────────────────────
    smem[si] = select(gpu_inf(), u[base + gidx], valid);

    // ── Load top halo (one global read per top-edge thread) ───────────────
    // Top-edge threads (local_r == 0) load the row immediately above this tile.
    // in_col guards columns that extend past the grid width.
    if local_r == 0u {
        let in_col = global_c < params.w;
        let val    = select(gpu_inf(),
                            select(gpu_inf(),
                                   u[base + (global_r - 1u) * params.w + global_c],
                                   global_r > 0u),
                            in_col);
        smem[smem_c] = val;                  // smem[0][smem_c]
    }

    // ── Load bottom halo ──────────────────────────────────────────────────
    if local_r == TILE - 1u {
        let below  = global_r + 1u;
        let in_col = global_c < params.w;
        let val    = select(gpu_inf(),
                            select(gpu_inf(),
                                   u[base + below * params.w + global_c],
                                   below < params.h),
                            in_col);
        smem[(TILE + 1u) * SMEM_W + smem_c] = val;  // smem[33][smem_c]
    }

    // ── Load left halo ────────────────────────────────────────────────────
    // Left-edge threads (local_c == 0): global_c == tile_col * TILE.
    // in_row guards rows that extend past the grid height.
    if local_c == 0u {
        let in_row = global_r < params.h;
        let val    = select(gpu_inf(),
                            select(gpu_inf(),
                                   u[base + global_r * params.w + (global_c - 1u)],
                                   global_c > 0u),
                            in_row);
        smem[smem_r * SMEM_W] = val;        // smem[smem_r][0]
    }

    // ── Load right halo ───────────────────────────────────────────────────
    if local_c == TILE - 1u {
        let right  = global_c + 1u;
        let in_row = global_r < params.h;
        let val    = select(gpu_inf(),
                            select(gpu_inf(),
                                   u[base + global_r * params.w + right],
                                   right < params.w),
                            in_row);
        smem[smem_r * SMEM_W + TILE + 1u] = val;  // smem[smem_r][33]
    }

    // All halo and own-cell loads must be visible before the iteration loop.
    workgroupBarrier();

    // ── TILE parallel update iterations — zero global reads ──────────────
    // smem_r ∈ [1,TILE], smem_c ∈ [1,TILE], so si±1 and si±SMEM_W are always
    // within [0,(TILE+2)²-1]: no bounds check needed in the inner loop.
    // Which neighbour this thread's cell can influence.  A neighbouring tile reads
    // our edge cells as its halo, so it only needs reprocessing if one of *those*
    // cells decreased — an interior change is invisible to it.  Bit 0 marks "this
    // tile changed at all" (so it may not have converged in TILE iterations and is
    // re-queued); bits 1..4 mark the four shared edges.
    var edge_mask = FLAG_SELF;
    if local_r == 0u        { edge_mask |= FLAG_TOP; }
    if local_r == TILE - 1u { edge_mask |= FLAG_BOTTOM; }
    if local_c == 0u        { edge_mask |= FLAG_LEFT; }
    if local_c == TILE - 1u { edge_mask |= FLAG_RIGHT; }

    // Accumulated in a register across iterations, then folded into the workgroup
    // atomic once — at most one atomic per thread instead of one per cell update.
    var my_flags = 0u;

    // Early exit: once an iteration changes nothing, the tile has converged and the
    // remaining iterations are pure waste.  `converged` gates only the *work* -- the
    // barrier still runs every iteration, because breaking out would leave the
    // remaining barriers in non-uniform control flow, which WGSL forbids.
    for (var iter: u32 = 0u; iter < ITERS; iter++) {
        if valid && !is_src && !is_wall {
            let a    = min(smem[si - 1u], smem[si + 1u]);
            let b    = min(smem[si - SMEM_W], smem[si + SMEM_W]);
            let cand = godunov(a, b, cost);
            if cand < smem[si] - params.conv_tol {
                smem[si] = cand;
                my_flags = edge_mask;
            }
        }
        workgroupBarrier();
    }

    if my_flags != 0u {
        atomicOr(&local_updated, my_flags);
    }

    // Write tile back to global memory.
    if valid {
        u[base + gidx] = smem[si];
    }
    storageBarrier();
    // local_updated must be complete before thread 0 reads it.
    workgroupBarrier();

    // Only thread 0 enqueues neighbours.
    if unit == 0u {
        let flags = atomicLoad(&local_updated);
        if flags != 0u {
            let next_round = current_round + 1u;
            let cap        = params.active_cap;
            try_enqueue(tile_slot, next_round, cap);
            if (flags & FLAG_TOP) != 0u && tile_row > 0u {
                try_enqueue(dest * params.num_tiles + (tile_row - 1u) * params.num_tile_cols + tile_col, next_round, cap);
            }
            if (flags & FLAG_BOTTOM) != 0u && tile_row + 1u < params.num_tile_rows {
                try_enqueue(dest * params.num_tiles + (tile_row + 1u) * params.num_tile_cols + tile_col, next_round, cap);
            }
            if (flags & FLAG_LEFT) != 0u && tile_col > 0u {
                try_enqueue(dest * params.num_tiles + tile_row * params.num_tile_cols + (tile_col - 1u), next_round, cap);
            }
            if (flags & FLAG_RIGHT) != 0u && tile_col + 1u < params.num_tile_cols {
                try_enqueue(dest * params.num_tiles + tile_row * params.num_tile_cols + (tile_col + 1u), next_round, cap);
            }
        }
    }
}
