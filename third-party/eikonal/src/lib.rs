pub mod fim;
mod fim_gpu;
mod fim_gpu_wgpu;
pub mod fmm;
pub mod fsm;
pub mod fsm_k4;

/// Set to `true` to print per-solver timing breakdowns; `false` for silent operation.
/// This is a compile-time constant: when `false` the dead branches are eliminated
/// by the compiler even in debug builds.
pub const PRINT_TIMINGS: bool = true;

/// Set to `true` to enable early termination in the GPU FIM solver: after each
/// 64-round batch the active-tile count is read back and, if zero, no further
/// batches are submitted.  Warm restarts with a small changed region converge in
/// far fewer rounds than `max_rounds` and benefit the most.
///
/// Set to `false` to always run all `max_rounds` batches — useful as a baseline
/// when benchmarking the early-termination overhead vs. the rounds it saves.
pub const GPU_EARLY_TERMINATION: bool = true;

/// Set to `true` to route the cold FIM batch path through the GPU (CubeCL/Metal).
/// Tiled FIM (Fu, Jeong, Pan, Kirby & Whitaker 2011) with async round loop:
///   - 16×16 tiles (256 threads/cube), 16 local Gauss-Seidel passes per launch
///   - pre-encoded max_rounds launches with a single GPU→CPU sync at the end
///
/// Performance: competitive with CPU FSM at K=1 on medium grids (≤1M cells),
/// but FSM wins for K>1 and for large grids because all tiles are dispatched
/// every round and GPU memory scales as K×N×4 bytes.
/// Default: false (USE_GPU_WGPU supersedes this path for production use).
pub const USE_GPU_BATCH: bool = false;

/// Set to `true` to route FIM batch solves through the wgpu direct path
/// (`fim_gpu_wgpu`). This path uses:
///   - `dispatch_workgroups_indirect`: only active tiles are dispatched each round
///   - single CommandEncoder per solve (one Metal command buffer = one GPU→CPU sync)
///   - persistent GPU buffers: `u_buf`, `speed_buf`, `tile_round_buf` survive between
///     calls so warm-start solves reuse the prior travel-time field on-GPU with no
///     CPU↔GPU roundtrip for the prior data
///
/// Takes priority over USE_GPU_BATCH for the two `_direct` batch entry points.
pub const USE_GPU_WGPU: bool = true;

#[cxx::bridge(namespace = "jupedsim::eikonal")]
mod ffi {
    extern "Rust" {
        /// Compute travel times using the Fast Sweeping Method (Zhao 2005).
        ///
        /// `speed_field`: one value per cell in row-major order; use 0.0 for obstacles.
        /// `sources`:     flat cell indices (row * width + col) where travel time = 0.
        /// `cell_size`:   physical size of one grid cell (metres).
        ///
        /// Returns a flat Vec<f64> of travel times in the same layout as the input.
        /// Unreachable cells retain f64::INFINITY.
        fn compute_travel_times(
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        ) -> Vec<f64>;

        /// Allocation-free FSM variant: writes travel times into a caller-supplied buffer.
        ///
        /// `out` must have length `width * height`. On entry all cells are initialised
        /// to f64::INFINITY by this function (the caller does not need to pre-fill).
        /// Use this form when calling from parallel threads to avoid per-call heap
        /// allocation contention.
        fn compute_travel_times_into(
            out: &mut [f64],
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Allocation-free FMM variant: same contract as `compute_travel_times_into`
        /// but uses the Fast Marching Method (O(N log N), more accurate on
        /// heterogeneous speed fields).
        fn compute_travel_times_fmm_into(
            out: &mut [f64],
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Allocation-free FIM variant: same contract as `compute_travel_times_into`
        /// but uses the Fast Iterative Method (converges in active-cell wavefront
        /// work proportional to the changed region, good for sparse sources).
        fn compute_travel_times_fim_into(
            out: &mut [f64],
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Parallel batch FSM: solve N point destinations concurrently via Rayon.
        ///
        /// `all_outs` is a flat buffer of `n_dests * width * height` doubles.
        /// Destination `i` receives its travel-time grid in the slice
        /// `all_outs[i*n .. (i+1)*n]` where `n = width * height`.
        /// `sources` must contain exactly one source cell index per destination.
        fn compute_travel_times_batch_fsm(
            all_outs: &mut [f64],
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Parallel batch FIM: same contract as `compute_travel_times_batch_fsm`.
        fn compute_travel_times_batch_fim(
            all_outs: &mut [f64],
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Parallel batch warm-start FIM: re-solve N destinations concurrently via Rayon.
        ///
        /// `all_outs` and `all_priors` are flat buffers of `n_dests * width * height` doubles.
        /// Destination `i` reads its prior from `all_priors[i*n .. (i+1)*n]` and writes its
        /// result into `all_outs[i*n .. (i+1)*n]`.
        /// `sources` contains one source cell index per destination; `changed_cells` is shared
        /// across all destinations (they use the same speed field, so the same cells changed).
        fn compute_travel_times_batch_fim_warm_into(
            all_outs: &mut [f64],
            speed_field: &[f64],
            all_priors: &[f64],
            changed_cells: &[u32],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Pointer-based parallel batch FSM: each Rayon task writes directly into the
        /// caller-supplied output buffer, eliminating the flat gather/scatter round-trip.
        ///
        /// `out_ptrs[i]` is a `*mut f64` cast to `usize`, pointing to exactly `n` f64 values.
        /// The caller must ensure all pointers are valid, non-aliasing, and live for the
        /// duration of this call. `n` must equal `width * height`.
        fn compute_travel_times_batch_fsm_direct(
            out_ptrs: &[usize],
            n: usize,
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Pointer-based parallel batch FIM: each Rayon task writes directly into the
        /// caller-supplied output buffer, eliminating the flat gather/scatter round-trip.
        ///
        /// `out_ptrs[i]` is a `*mut f64` cast to `usize`, pointing to exactly `n` f64 values.
        /// The caller must ensure all pointers are valid, non-aliasing, and live for the
        /// duration of this call. `n` must equal `width * height`.
        fn compute_travel_times_batch_fim_direct(
            out_ptrs: &[usize],
            n: usize,
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Pointer-based parallel batch warm-start FIM: same as `compute_travel_times_batch_fim_direct`
        /// but warm-starts each solve from the buffer pointed to by `prior_ptrs[i]`.
        ///
        /// Eliminates both the prior-gather flat buffer and the result-scatter flat buffer.
        /// `out_ptrs` and `prior_ptrs` must not alias each other or the speed field.
        fn compute_travel_times_batch_fim_warm_direct(
            out_ptrs: &[usize],
            prior_ptrs: &[usize],
            n: usize,
            speed_field: &[f64],
            changed_cells: &[u32],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// K=4 parallel FSM: each Rayon task solves four destinations simultaneously,
        /// sharing one speed-field read per cell per sweep.
        ///
        /// `all_outs`: length must be `sources.len() / 4 * width * height`.
        /// Laid out as contiguous per-destination grids:
        ///   group g, lane k → `all_outs[(g*4 + k) * n .. (g*4 + k + 1) * n]`.
        /// `sources`: length must be a multiple of 4; pad unused lanes with `u32::MAX`.
        fn compute_travel_times_batch_fsm_k4(
            all_outs: &mut [f64],
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );

        /// Compute travel times using the Fast Marching Method (Sethian 1996).
        ///
        /// Same interface as `compute_travel_times`. FMM processes cells in
        /// strictly increasing order of travel time (O(N log N)), which gives
        /// better accuracy on strongly heterogeneous speed fields compared to FSM.
        fn compute_travel_times_fmm(
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        ) -> Vec<f64>;

        /// Compute travel times using the Fast Iterative Method (Jeong & Whitaker 2008).
        ///
        /// Cold-start variant: same interface as FSM/FMM. Prefer `compute_travel_times_fim_warm`
        /// when a prior solution is available.
        fn compute_travel_times_fim(
            speed_field: &[f64],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        ) -> Vec<f64>;

        /// Warm-start FIM: re-solves using `prior` (a previous travel-time field) as the
        /// initial guess. `changed_cells` contains the flat cell indices where the speed
        /// field changed since `prior` was computed; those cells and their neighbours are
        /// seeded into the active list. Converges in O(K) active cell work where K is the
        /// size of the changed region. The caller is responsible for providing the correct
        /// changed-cell set (computed cheaply during speed-field rebuild in C++).
        fn compute_travel_times_fim_warm(
            speed_field: &[f64],
            prior: &[f64],
            changed_cells: &[u32],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        ) -> Vec<f64>;

        /// Allocation-free warm-start FIM: copies `prior` into `out` then re-solves in
        /// place. Eliminates the Rust-side Vec allocation and the subsequent C++ copy.
        /// `out`, `prior`, and `speed_field` must all have length `width * height`.
        fn compute_travel_times_fim_warm_into(
            out: &mut [f64],
            speed_field: &[f64],
            prior: &[f64],
            changed_cells: &[u32],
            sources: &[u32],
            width: usize,
            height: usize,
            cell_size: f64,
        );
    }
}

fn compute_travel_times(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    fsm::solve(speed_field, sources, width, height, cell_size)
}

fn compute_travel_times_into(
    out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        out.len(),
        width * height,
        "out length must equal width * height"
    );
    fsm::solve_into(out, speed_field, sources, width, height, cell_size);
}

fn compute_travel_times_fmm_into(
    out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        out.len(),
        width * height,
        "out length must equal width * height"
    );
    fmm::solve_into(out, speed_field, sources, width, height, cell_size);
}

fn compute_travel_times_fim_into(
    out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        out.len(),
        width * height,
        "out length must equal width * height"
    );
    fim::solve_into(out, speed_field, sources, width, height, cell_size);
}

fn compute_travel_times_fmm(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    fmm::solve(speed_field, sources, width, height, cell_size)
}

fn compute_travel_times_fim(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    fim::solve(speed_field, sources, width, height, cell_size)
}

fn compute_travel_times_batch_fsm(
    all_outs: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    use rayon::prelude::*;
    let n = width * height;
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        all_outs.len(),
        n * sources.len(),
        "all_outs length must equal n_dests * width * height"
    );
    all_outs
        .par_chunks_exact_mut(n)
        .zip(sources.par_iter())
        .for_each(|(out, &src)| {
            fsm::solve_into(out, speed_field, &[src], width, height, cell_size);
        });
}

fn compute_travel_times_batch_fim(
    all_outs: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    use rayon::prelude::*;
    let n = width * height;
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        all_outs.len(),
        n * sources.len(),
        "all_outs length must equal n_dests * width * height"
    );
    all_outs
        .par_chunks_exact_mut(n)
        .zip(sources.par_iter())
        .for_each(|(out, &src)| {
            fim::solve_into(out, speed_field, &[src], width, height, cell_size);
        });
}

fn compute_travel_times_batch_fim_warm_into(
    all_outs: &mut [f64],
    speed_field: &[f64],
    all_priors: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    use rayon::prelude::*;
    let n = width * height;
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        all_outs.len(),
        n * sources.len(),
        "all_outs length must equal n_dests * width * height"
    );
    assert_eq!(
        all_priors.len(),
        n * sources.len(),
        "all_priors length must equal n_dests * width * height"
    );
    all_outs
        .par_chunks_exact_mut(n)
        .zip(all_priors.par_chunks_exact(n))
        .zip(sources.par_iter())
        .for_each(|((out, prior), &src)| {
            fim::solve_warm_into(
                out,
                speed_field,
                prior,
                changed_cells,
                &[src],
                width,
                height,
                cell_size,
            );
        });
}

fn compute_travel_times_batch_fsm_direct(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    use rayon::prelude::*;
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        out_ptrs.len(),
        sources.len(),
        "out_ptrs and sources must have the same length"
    );
    // Safety: caller guarantees each out_ptr is a valid, n-element, non-aliasing
    // *mut f64 buffer that lives for the duration of this call.
    out_ptrs
        .par_iter()
        .zip(sources.par_iter())
        .for_each(|(&raw, &src)| {
            let out = unsafe { std::slice::from_raw_parts_mut(raw as *mut f64, n) };
            fsm::solve_into(out, speed_field, &[src], width, height, cell_size);
        });
}

fn compute_travel_times_batch_fim_direct(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        out_ptrs.len(),
        sources.len(),
        "out_ptrs and sources must have the same length"
    );

    if USE_GPU_WGPU && fim_gpu_wgpu::fits_in_gpu(out_ptrs.len(), n) {
        fim_gpu_wgpu::solve_cold(out_ptrs, n, speed_field, sources, width, height, cell_size);
        return;
    }
    if USE_GPU_BATCH {
        fim_gpu::solve_batch_direct(out_ptrs, n, speed_field, sources, width, height, cell_size);
        return;
    }

    // CPU Rayon path: one Rayon task per destination, serial FIM within each.
    use rayon::prelude::*;
    // Safety: the caller (C++) guarantees each out_ptr is a valid, n-element, non-aliasing
    // *mut f64 buffer that lives for the duration of this call.
    out_ptrs
        .par_iter()
        .zip(sources.par_iter())
        .for_each(|(&raw, &src)| {
            let out = unsafe { std::slice::from_raw_parts_mut(raw as *mut f64, n) };
            fim::solve_into(out, speed_field, &[src], width, height, cell_size);
        });
}

fn compute_travel_times_batch_fim_warm_direct(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    if USE_GPU_WGPU && fim_gpu_wgpu::fits_in_gpu(out_ptrs.len(), n) {
        fim_gpu_wgpu::solve_warm(
            out_ptrs,
            prior_ptrs,
            n,
            speed_field,
            changed_cells,
            sources,
            width,
            height,
            cell_size,
        );
        return;
    }

    use rayon::prelude::*;
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        out_ptrs.len(),
        sources.len(),
        "out_ptrs and sources must have the same length"
    );
    assert_eq!(
        prior_ptrs.len(),
        sources.len(),
        "prior_ptrs and sources must have the same length"
    );
    // Safety: caller guarantees each pair of pointers is valid, n-element, and non-aliasing
    // with each other and the speed field for the duration of this call.
    out_ptrs
        .par_iter()
        .zip(prior_ptrs.par_iter())
        .zip(sources.par_iter())
        .for_each(|((&out_raw, &prior_raw), &src)| {
            let out = unsafe { std::slice::from_raw_parts_mut(out_raw as *mut f64, n) };
            let prior = unsafe { std::slice::from_raw_parts(prior_raw as *const f64, n) };
            fim::solve_warm_into(
                out,
                speed_field,
                prior,
                changed_cells,
                &[src],
                width,
                height,
                cell_size,
            );
        });
}

fn compute_travel_times_batch_fsm_k4(
    all_outs: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    use rayon::prelude::*;
    let n = width * height;
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        sources.len() % 4,
        0,
        "sources length must be a multiple of 4"
    );
    assert_eq!(
        all_outs.len(),
        sources.len() * n,
        "all_outs length must equal sources.len() * width * height"
    );
    // Each Rayon task gets one group of 4 destinations and processes them with a single
    // K=4 interleaved sweep — one speed-field read per cell serves all four updates.
    all_outs
        .par_chunks_exact_mut(4 * n)
        .zip(sources.par_chunks_exact(4))
        .for_each(|(group_out, group_src)| {
            let src4: &[u32; 4] = group_src.try_into().unwrap();
            fsm_k4::solve_k4_into(group_out, speed_field, src4, width, height, cell_size);
        });
}

fn compute_travel_times_fim_warm_into(
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
    assert_eq!(
        speed_field.len(),
        n,
        "speed_field length must equal width * height"
    );
    assert_eq!(prior.len(), n, "prior length must equal width * height");
    assert_eq!(out.len(), n, "out length must equal width * height");
    fim::solve_warm_into(
        out,
        speed_field,
        prior,
        changed_cells,
        sources,
        width,
        height,
        cell_size,
    );
}

fn compute_travel_times_fim_warm(
    speed_field: &[f64],
    prior: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    assert_eq!(
        speed_field.len(),
        width * height,
        "speed_field length must equal width * height"
    );
    assert_eq!(
        prior.len(),
        width * height,
        "prior length must equal width * height"
    );
    fim::solve_warm(
        speed_field,
        prior,
        changed_cells,
        sources,
        width,
        height,
        cell_size,
    )
}

// ── Multi-source batch entry points ──────────────────────────────────────────
//
// CSR format: destination `i` uses `sources_flat[src_offsets[i]..src_offsets[i+1]]`.
// `src_offsets` has length `out_ptrs.len() + 1`.

pub fn fim_batch_cold_ms(
    out_ptrs: &[usize],
    n: usize,
    speed: &[f64],
    sources_flat: &[u32],
    src_offsets: &[u32],
    w: usize,
    h: usize,
    cs: f64,
) {
    use rayon::prelude::*;
    let n_dests = out_ptrs.len();
    debug_assert_eq!(src_offsets.len(), n_dests + 1);

    if USE_GPU_WGPU && fim_gpu_wgpu::fits_in_gpu(n_dests, n) {
        fim_gpu_wgpu::solve_cold_ms(out_ptrs, n, speed, sources_flat, src_offsets, w, h, cs);
        return;
    }
    (0..n_dests).into_par_iter().for_each(|i| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptrs[i] as *mut f64, n) };
        fim::solve_into(
            out,
            speed,
            &sources_flat[src_offsets[i] as usize..src_offsets[i + 1] as usize],
            w,
            h,
            cs,
        );
    });
}

pub fn fim_batch_warm_ms(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed: &[f64],
    changed: &[u32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    w: usize,
    h: usize,
    cs: f64,
) {
    use rayon::prelude::*;
    let n_dests = out_ptrs.len();
    debug_assert_eq!(src_offsets.len(), n_dests + 1);

    if USE_GPU_WGPU && fim_gpu_wgpu::fits_in_gpu(n_dests, n) {
        println!("GPU warm FIM batch path");
        fim_gpu_wgpu::solve_warm_ms(
            out_ptrs,
            prior_ptrs,
            n,
            speed,
            changed,
            sources_flat,
            src_offsets,
            w,
            h,
            cs,
        );
        return;
    }
    (0..n_dests).into_par_iter().for_each(|i| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptrs[i] as *mut f64, n) };
        let prior = unsafe { std::slice::from_raw_parts(prior_ptrs[i] as *const f64, n) };
        fim::solve_warm_into(
            out,
            speed,
            prior,
            changed,
            &sources_flat[src_offsets[i] as usize..src_offsets[i + 1] as usize],
            w,
            h,
            cs,
        );
    });
}

pub fn fim_batch_cold_ms_f32(
    out_ptrs: &[usize],
    n: usize,
    speed: &[f32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    w: usize,
    h: usize,
    cs: f32,
) {
    use rayon::prelude::*;
    let n_dests = out_ptrs.len();
    debug_assert_eq!(src_offsets.len(), n_dests + 1);

    if USE_GPU_WGPU && fim_gpu_wgpu::fits_in_gpu(n_dests, n) {
        fim_gpu_wgpu::solve_cold_ms_f32(out_ptrs, n, speed, sources_flat, src_offsets, w, h, cs);
        return;
    }
    (0..n_dests).into_par_iter().for_each(|i| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptrs[i] as *mut f32, n) };
        fsm::solve_into_typed(
            out,
            speed,
            &sources_flat[src_offsets[i] as usize..src_offsets[i + 1] as usize],
            w,
            h,
            cs,
        );
    });
}

pub fn fim_batch_warm_ms_f32(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed: &[f32],
    changed: &[u32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    w: usize,
    h: usize,
    cs: f32,
) {
    use rayon::prelude::*;
    let n_dests = out_ptrs.len();
    debug_assert_eq!(src_offsets.len(), n_dests + 1);

    if USE_GPU_WGPU && fim_gpu_wgpu::fits_in_gpu(n_dests, n) {
        println!("GPU warm FIM batch path FP32");
        fim_gpu_wgpu::solve_warm_ms_f32(
            out_ptrs,
            n,
            speed,
            changed,
            sources_flat,
            src_offsets,
            w,
            h,
            cs,
        );
        return;
    }
    // CPU: cold re-solve with FSM (prior_ptrs unused — f32 has no CPU warm FIM).
    let _ = prior_ptrs;
    (0..n_dests).into_par_iter().for_each(|i| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptrs[i] as *mut f32, n) };
        println!("CPU warm FSM batch path FP32");
        fsm::solve_into_typed(
            out,
            speed,
            &sources_flat[src_offsets[i] as usize..src_offsets[i + 1] as usize],
            w,
            h,
            cs,
        );
    });
}
