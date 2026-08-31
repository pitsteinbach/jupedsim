#pragma once
// Hand-written header for the cxx-generated bridge.
// The actual bridge header is emitted into the build tree by cxx;
// this file is the stable include path for JuPedSim to use.
#include "rust/cxx.h"

namespace jupedsim::eikonal
{

/// Compute travel times from exit cells on a 2D grid.
/// speed_field: row-major, 0.0 marks obstacles.
/// sources:     flat indices (row * width + col) of exit cells.
/// cell_size:   physical size of one grid cell in metres.
/// Returns travel times in the same layout; unreachable cells are infinity.
rust::Vec<double> compute_travel_times(
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

rust::Vec<double> compute_travel_times_fmm(
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

rust::Vec<double> compute_travel_times_fim(
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

rust::Vec<double> compute_travel_times_fim_warm(
    rust::Slice<const double> speed_field,
    rust::Slice<const double> prior,
    rust::Slice<const uint32_t> changed_cells,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Allocation-free warm-start FIM: copies `prior` into `out` then re-solves in place.
/// Eliminates the Rust-side Vec allocation and the subsequent C++ copy of the result.
/// `out`, `prior`, and `speed_field` must each have length `width * height`.
void compute_travel_times_fim_warm_into(
    rust::Slice<double> out,
    rust::Slice<const double> speed_field,
    rust::Slice<const double> prior,
    rust::Slice<const uint32_t> changed_cells,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Allocation-free FSM variant: writes results into a caller-supplied buffer.
/// `out` must have length width*height. Initialised to infinity internally.
/// Use from parallel threads to avoid per-call heap allocation contention.
void compute_travel_times_into(
    rust::Slice<double> out,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Allocation-free FMM variant: same contract as compute_travel_times_into
/// but uses the Fast Marching Method (O(N log N), more accurate on
/// heterogeneous speed fields, priority-queue overhead).
void compute_travel_times_fmm_into(
    rust::Slice<double> out,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Allocation-free FIM variant: same contract as compute_travel_times_into
/// but uses the Fast Iterative Method (active-wavefront propagation,
/// auxiliary scratch still allocated per call).
void compute_travel_times_fim_into(
    rust::Slice<double> out,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Parallel batch FSM: solve N point destinations concurrently via Rayon.
/// `all_outs` must have length n_dests * width * height. Destination i writes
/// its travel-time grid into all_outs[i*n .. (i+1)*n] where n = width * height.
/// `sources` contains exactly one source cell index per destination (len == n_dests).
void compute_travel_times_batch_fsm(
    rust::Slice<double> all_outs,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Parallel batch FIM: same contract as compute_travel_times_batch_fsm.
void compute_travel_times_batch_fim(
    rust::Slice<double> all_outs,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Parallel batch warm-start FIM: re-solve N destinations concurrently via Rayon.
/// `all_outs` and `all_priors` must each have length n_dests * width * height.
/// Destination i reads prior from all_priors[i*n..(i+1)*n] and writes into all_outs[i*n..(i+1)*n].
/// `changed_cells` is shared across all destinations (same speed field, same changed region).
void compute_travel_times_batch_fim_warm_into(
    rust::Slice<double> all_outs,
    rust::Slice<const double> speed_field,
    rust::Slice<const double> all_priors,
    rust::Slice<const uint32_t> changed_cells,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Pointer-based parallel batch FSM: each Rayon task writes directly into the
/// caller-supplied buffer, eliminating the flat allOuts round-trip.
/// out_ptrs[i] is a double* cast to size_t, pointing to exactly n = width*height doubles.
/// All pointed-to buffers must be valid, non-aliasing, and live for the duration of the call.
void compute_travel_times_batch_fsm_direct(
    rust::Slice<const size_t> out_ptrs,
    size_t n,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Pointer-based parallel batch FIM: each Rayon task writes directly into the
/// caller-supplied buffer, eliminating the flat allOuts round-trip.
/// out_ptrs[i] is a double* cast to size_t, pointing to exactly n = width*height doubles.
/// All pointed-to buffers must be valid, non-aliasing, and live for the duration of the call.
void compute_travel_times_batch_fim_direct(
    rust::Slice<const size_t> out_ptrs,
    size_t n,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// Pointer-based parallel batch warm-start FIM: same as compute_travel_times_batch_fim_direct
/// but warm-starts each solve from the buffer at prior_ptrs[i].
/// Eliminates both the prior-gather flat buffer and the result-scatter flat buffer.
void compute_travel_times_batch_fim_warm_direct(
    rust::Slice<const size_t> out_ptrs,
    rust::Slice<const size_t> prior_ptrs,
    size_t n,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> changed_cells,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

/// K=4 parallel FSM: each Rayon task solves four destinations simultaneously,
/// sharing one speed-field read per cell per sweep.
///
/// sources.size() must be a multiple of 4; pad unused lanes with UINT32_MAX.
/// all_outs.size() must equal sources.size() * width * height.
/// Output layout: group g, lane k → all_outs[(g*4 + k) * n .. (g*4 + k + 1) * n].
void compute_travel_times_batch_fsm_k4(
    rust::Slice<double> all_outs,
    rust::Slice<const double> speed_field,
    rust::Slice<const uint32_t> sources,
    size_t width,
    size_t height,
    double cell_size);

} // namespace jupedsim::eikonal
