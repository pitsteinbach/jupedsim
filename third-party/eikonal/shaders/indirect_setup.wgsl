// Single-thread kernel that:
//   1. Reads next_active_count from active_count
//   2. Writes it to indirect_buf[0] (x dispatch count)
//   3. Sets indirect_buf[1] = 1, indirect_buf[2] = 1 (y, z always 1)
//   4. Resets active_count to 0 for the next ping-pong write

@group(0) @binding(0) var<storage, read_write> indirect_buf:  array<u32>;
@group(0) @binding(1) var<storage, read_write> active_count:  atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn setup_indirect() {
    let count       = atomicExchange(&active_count, 0u);
    indirect_buf[0] = count;
    indirect_buf[1] = 1u;
    indirect_buf[2] = 1u;
}
