// Single-thread kernel that:
//   1. Reads next_active_count from active_count
//   2. Publishes it to round_count[0] so fim_update can bound-check its workgroups
//   3. Writes a 2-D dispatch size into indirect_buf that covers `count` workgroups
//      without exceeding max_compute_workgroups_per_dimension in any dimension
//   4. Resets active_count to 0 for the next ping-pong write
//
// Why 2-D: the active-tile count is summed across all K destinations, so it grows
// with K.  A 1-D dispatch of `count` workgroups silently exceeds the 65 535
// per-dimension limit once K is large (measured: k=16 on a 4107x3769 grid peaks
// well past the limit), and every tile past the limit is then never dispatched.
// Because try_enqueue has already bumped those tiles' tile_round tag, they can
// never be re-enqueued either — the wavefront dies and the solve returns a
// partially-filled field.  Splitting across x and y keeps every dimension legal.
//
// The grid is rounded up, so the last row of workgroups can overshoot `count`.
// fim_update discards the overshoot using round_count[0].

@group(0) @binding(0) var<storage, read_write> indirect_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> active_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> round_count:  array<u32>;

// WebGPU guarantees max_compute_workgroups_per_dimension >= 65535, so this width
// is always legal regardless of the adapter's reported limit.
const MAX_WG_PER_DIM: u32 = 65535u;

@compute @workgroup_size(1, 1, 1)
fn setup_indirect() {
    let count = atomicExchange(&active_count, 0u);
    round_count[0] = count;

    // count == 0 yields (0, 0, 1): a no-op dispatch, which is what we want.
    indirect_buf[0] = min(count, MAX_WG_PER_DIM);
    indirect_buf[1] = (count + MAX_WG_PER_DIM - 1u) / MAX_WG_PER_DIM;
    indirect_buf[2] = 1u;
}
