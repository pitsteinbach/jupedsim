//! Stateful GPU eikonal solver using wgpu directly (bypasses CubeCL).
//!
//! Key design choices vs. the CubeCL path:
//!   - `dispatch_workgroups_indirect`: only active tiles are dispatched; settled
//!     tiles cost nothing instead of immediately terminating a dummy workgroup.
//!   - Single command encoder per solve: all rounds are pre-encoded into one
//!     Metal command buffer, so there is exactly ONE GPU→CPU sync per solve.
//!   - Persistent `u_buf`: the travel-time grid survives between calls, enabling
//!     true warm starts where only the changed region is re-solved.
//!
//! # Memory lifecycle
//!
//! `GpuContext` (device, queue, pipelines) is created once and lives for the
//! process lifetime. `GpuFimSolver` (dimension-specific buffers and bind groups)
//! is rebuilt when the grid dimensions or destination count change.  Old buffers
//! are explicitly dropped and the device is polled before new ones are allocated,
//! so Metal's allocator sees the freed pages before it has to satisfy the next
//! request.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Tile edge length in cells — THE single definition. Everything else is derived:
/// the shaders receive it through `wgsl_prelude()`, which also supplies SMEM_W,
/// SMEM_LEN and ITERS, and their workgroup_size and smem array are written in terms
/// of those. Changing this one line is sufficient; nothing else needs editing.
///
/// Constraints: TILE*TILE threads must fit a workgroup (Metal caps at 1024, so
/// TILE <= 32), and (TILE+2)^2 floats must fit workgroup storage.
const TILE: usize = 8;

/// WGSL constants injected at the top of every shader module so that TILE above is
/// the only place a tile dimension is written down.
///
/// ITERS is derived, not tuned: the halo is snapshotted at entry and never
/// refreshed, so the iteration loop solves a fixed boundary-value problem on the
/// interior, and one Jacobi iteration moves influence exactly one cell of Manhattan
/// distance. Fully relaxing the tile therefore takes its 4-connected graph
/// diameter, TILE steps down plus TILE-1 across. See the header of fim_update.wgsl
/// for the measurements behind this.
fn wgsl_prelude() -> String {
    format!(
        "// ---- generated from TILE in fim_gpu_wgpu.rs; do not edit in .wgsl ----\n\
         const TILE:     u32 = {tile}u;\n\
         const SMEM_W:   u32 = TILE + 2u;\n\
         const SMEM_LEN: u32 = SMEM_W * SMEM_W;\n\
         const ITERS:    u32 = 2u * TILE - 1u;\n\
         // ---------------------------------------------------------------\n",
        tile = TILE
    )
}

/// Shader source with the generated prelude prepended.
fn wgsl(body: &str) -> String {
    let mut s = wgsl_prelude();
    s.push_str(body);
    s
}

/// Per-dimension workgroup ceiling. WebGPU guarantees
/// `max_compute_workgroups_per_dimension >= 65535`, so splitting a dispatch at
/// this width is legal on every adapter. Any dispatch whose extent scales with
/// `k` or `n` must be split across x/y rather than issued as a flat 1-D grid —
/// exceeding the limit silently drops the tail of the grid.
const MAX_WG_PER_DIM: u32 = 65535;

/// Split a linear workgroup count into a 2-D grid that stays within
/// `MAX_WG_PER_DIM` per dimension. The grid rounds up, so the shader must
/// bound-check the flat index against the true count.
fn split_dispatch(count: u32) -> (u32, u32) {
    (count.min(MAX_WG_PER_DIM), count.div_ceil(MAX_WG_PER_DIM))
}

// ── Deferred readback token ────────────────────────────────────────────────────
// Produced by `dispatch_*` functions; consumed by `try_collect_pending`.
// `map_async` is issued immediately after the blit submit so that the single
// `poll(Wait)` at collect time both completes the blit and fires the map
// callback — identical to the blocking path but deferred across iterations.
struct PendingReadback {
    /// `None` when u_buf was mapped directly: no new work was submitted, so the
    /// collector waits on all outstanding submissions instead of a specific one.
    submission: Option<wgpu::SubmissionIndex>,
    out_ptrs: Vec<usize>,
    is_f32: bool,
}
const CONV_TOL: f32 = 1e-2;
// fill_sentinel.wgsl initialises u_buf with bitcast<f32>(0x7F800000u) = f32::INFINITY.
// f32::INFINITY as f64 == f64::INFINITY per IEEE 754, so neither readback path needs
// a per-element conditional — readback_f32 is a plain copy_from_slice per destination,
// and readback (f64) is a straight widening cast with no branch.

// ── GpuParams — must be 48 bytes, 16-byte aligned (std140 uniform layout) ────

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuParams {
    w: u32,
    h: u32,
    k: u32,
    num_tile_cols: u32, // row 0 — 16 bytes
    num_tile_rows: u32,
    num_tiles: u32,
    n: u32,
    cell_size: f32, // row 1 — 16 bytes
    conv_tol: f32,
    active_cap: u32,
    is_warm: u32, // 0 = cold (one-directional guard), 1 = warm (bidirectional)
    _pad: u32,    // row 2 — 16 bytes; total 48 bytes
}

unsafe fn as_bytes<T: Sized>(t: &T) -> &[u8] {
    std::slice::from_raw_parts(t as *const T as *const u8, std::mem::size_of::<T>())
}

// Zero-copy byte views — avoids allocating a second 100 MB buffer just to
// reinterpret the data type.  Alignment is fine: f32/u32 align to 4 bytes,
// and [u8] only requires 1-byte alignment.
fn f32_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) }
}

fn u32_as_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) }
}

// Selects the FIM update shader at compile time.
// Default (no feature): 18×18 smem with 1-cell halo — border reads happen once.
// `no_halo` feature: 16×16 smem, border neighbours read from global each iteration.
#[cfg(not(feature = "no_halo"))]
const FIM_SHADER_SRC: &str = include_str!("../shaders/fim_update.wgsl");
#[cfg(feature = "no_halo")]
const FIM_SHADER_SRC: &str = include_str!("../shaders/fim_update_no_halo.wgsl");

// ── Device-level context (created once, shared across all solver instances) ───

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// True when `u_buf` carries MAP_READ and can be read without a staging blit.
    mappable: bool,
    fim_pipeline: wgpu::ComputePipeline,
    setup_pipeline: wgpu::ComputePipeline,
    fill_pipeline: wgpu::ComputePipeline,
    reset_pipeline: wgpu::ComputePipeline,
    threshold_pipeline: wgpu::ComputePipeline,
    boundary_pipeline: wgpu::ComputePipeline,
    boundary_bgl: wgpu::BindGroupLayout,
    threshold_bgl: wgpu::BindGroupLayout,
    fim_bgl_0: wgpu::BindGroupLayout,
    fim_bgl_1: wgpu::BindGroupLayout,
    setup_bgl_0: wgpu::BindGroupLayout,
    fill_bgl_0: wgpu::BindGroupLayout,
    reset_bgl_0: wgpu::BindGroupLayout,
}

impl GpuContext {
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no GPU adapter found");

        // Request the device with the adapter's own native limits instead of
        // Limits::default() (which caps max_storage_buffer_binding_size at the
        // WebGPU portability baseline of 128 MB).  On Metal/Apple this can be
        // gigabytes — the full GPU address space.
        // On unified memory u_buf can be mapped directly, so results are read in
        // place instead of being blitted into a second full-size staging buffer
        // first. wgpu normally forbids MAP_READ | STORAGE because on a discrete
        // GPU it would force the storage buffer into system memory — hence the
        // extra IntegratedGpu check, which keeps the staging path on dGPUs.
        // Measured saving at k=16 on 4107x3769: ~115 ms of blit and 991 MB.
        let native_limits = adapter.limits();
        let mappable = adapter
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
            && adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu
            // Escape hatch: EIKONAL_NO_MAPPABLE=1 forces the staging path, for
            // A/B measurement and as a fallback if in-place mapping misbehaves.
            && std::env::var("EIKONAL_NO_MAPPABLE").is_err();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: if mappable {
                wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
            } else {
                wgpu::Features::empty()
            },
            required_limits: native_limits,
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .expect("failed to open GPU device");

        // ── Shaders ───────────────────────────────────────────────────────────
        let fim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl(FIM_SHADER_SRC).into()),
        });
        let setup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl(include_str!("../shaders/indirect_setup.wgsl")).into()),
        });
        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl(include_str!("../shaders/fill_sentinel.wgsl")).into()),
        });
        let reset_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl(include_str!("../shaders/reset_tiles.wgsl")).into()),
        });

        // ── Bind-group layouts ────────────────────────────────────────────────
        // FIM group 0: params(uniform) u(rw) speed(r) sources(r) tile_round(rw)
        let fim_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(1, storage_rw()),
                bgl_entry(2, storage_r()),
                bgl_entry(3, storage_r()),
                bgl_entry(4, storage_rw()),
                bgl_entry(5, storage_r()),
            ],
        });
        // FIM group 1: active_in(r) active_out(rw) active_out_count(rw)
        let fim_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, storage_r()),
                bgl_entry(1, storage_rw()),
                bgl_entry(2, storage_rw()),
            ],
        });
        // Setup group 0: indirect_buf(rw) active_count(rw)
        let setup_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, storage_rw()),
                bgl_entry(1, storage_rw()),
                bgl_entry(2, storage_rw()),
            ],
        });
        // Fill group 0: u(rw) — sentinel fill shader
        let fill_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_entry(0, storage_rw())],
        });
        // Reset group 0: params(uniform) u(rw) tile_slots(r) — dirty-tile reset
        let reset_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(1, storage_rw()),
                bgl_entry(2, storage_r()),
                bgl_entry(3, storage_r()),
            ],
        });

        // ── Pipelines ─────────────────────────────────────────────────────────
        let fim_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&fim_bgl_0), Some(&fim_bgl_1)],
            immediate_size: 0,
        });
        let setup_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&setup_bgl_0)],
            immediate_size: 0,
        });
        let fill_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&fill_bgl_0)],
            immediate_size: 0,
        });
        let reset_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&reset_bgl_0)],
            immediate_size: 0,
        });

        let fim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&fim_layout),
            module: &fim_shader,
            entry_point: Some("fim_update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let setup_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&setup_layout),
            module: &setup_shader,
            entry_point: Some("setup_indirect"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&fill_layout),
            module: &fill_shader,
            entry_point: Some("fill_sentinel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        // Boundary seeding: starts the warm wave along the whole threshold contour.
        let boundary_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl(include_str!("../shaders/seed_boundary.wgsl")).into()),
        });
        let boundary_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(1, storage_r()),
                bgl_entry(2, storage_rw()),
                bgl_entry(3, storage_rw()),
                bgl_entry(4, storage_rw()),
            ],
        });
        let boundary_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&boundary_bgl)],
            immediate_size: 0,
        });
        let boundary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&boundary_layout),
            module: &boundary_shader,
            entry_point: Some("seed_boundary"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Threshold reset: the correctness mechanism for warm restarts.
        let threshold_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl(include_str!("../shaders/threshold_reset.wgsl")).into()),
        });
        let threshold_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(1, storage_rw()),
                bgl_entry(2, storage_r()),
            ],
        });
        let threshold_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&threshold_bgl)],
            immediate_size: 0,
        });
        let threshold_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&threshold_layout),
            module: &threshold_shader,
            entry_point: Some("threshold_reset"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&reset_layout),
            module: &reset_shader,
            entry_point: Some("reset_dirty_tiles"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuContext {
            device,
            queue,
            mappable,
            fim_pipeline,
            setup_pipeline,
            fill_pipeline,
            reset_pipeline,
            threshold_pipeline,
            boundary_pipeline,
            boundary_bgl,
            threshold_bgl,
            fim_bgl_0,
            fim_bgl_1,
            setup_bgl_0,
            fill_bgl_0,
            reset_bgl_0,
        }
    }
}

// ── Dimension-specific buffers and bind groups ────────────────────────────────

#[allow(dead_code)] // buffers kept alive through this struct; accessed via bind groups
struct GpuFimSolver {
    // Persistent GPU buffers (survive between solves — warm start key)
    u_buf: wgpu::Buffer,          // K × N  f32
    speed_buf: wgpu::Buffer,      // N      f32
    sources_buf: wgpu::Buffer,    // K      u32
    tile_round_buf: wgpu::Buffer, // K × num_tiles  atomic<u32>

    // Active-tile ping-pong
    active_a: wgpu::Buffer,
    active_b: wgpu::Buffer,
    count_a: wgpu::Buffer,
    count_b: wgpu::Buffer,

    indirect_buf: wgpu::Buffer,
    round_count_buf: wgpu::Buffer,
    threshold_buf: wgpu::Buffer,
    thresholds_buf: wgpu::Buffer,
    // 4-byte MAP_READ buffer: receives indirect_buf[0] after each batch to check
    // whether any tiles remain active. Zero → FIM has converged, stop early.
    count_staging: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    // Single MAP_READ staging buffer covering all k destinations (k × N × 4 bytes).
    // The entire u_buf is blitted in one command, mapped with one map_async issued
    // before the poll, so one Wait poll both completes the GPU copy and fires the
    // map callback — no per-destination allocation, map, or unmap overhead.
    /// Full-size MAP_READ copy target. `None` when `ctx.mappable`, i.e. when
    /// `u_buf` is mapped directly and no staging copy exists at all.
    staging: Option<wgpu::Buffer>,

    // Bind groups
    bg_persistent: wgpu::BindGroup,
    bg_ping: wgpu::BindGroup,
    bg_pong: wgpu::BindGroup,
    setup_bg_ping: wgpu::BindGroup,
    setup_bg_pong: wgpu::BindGroup,
    fill_bg: wgpu::BindGroup,

    // Reusable scratch buffer for f64→f32 conversion (speed field and priors).
    // Pre-allocated in new() so upload_speed_and_sources never heap-allocates.
    speed_scratch: Vec<f32>,

    // Dims
    pub k: usize,
    pub n: usize,
    pub width: usize,
    pub height: usize,
    tile_cols: usize,
    tile_rows: usize,
    num_tiles: usize,
    max_rounds: u32,
    active_cap: u32,
    // True after any completed cold solve on this instance. False on fresh
    // allocation (k/n/grid change). Warm entry points fall back to a full cold
    // init when this is false, preventing FIM from being stuck on an all-zero
    // or uninitialized u_buf.
    pub u_buf_valid: bool,
    // Sources used in the last solve, in CSR format (for multi-source) or as a
    // flat per-slot array (for single-source, prev_src_offsets is empty).
    // Used to detect slot→destination mismatches across warm calls: if the
    // destination assigned to GPU slot d differs from last time, the prior in
    // u_buf[d*n..] is for the wrong destination and must be replaced from CPU.
    prev_sources_flat: Vec<u32>,
    prev_src_offsets: Vec<u32>,
    // Deferred readback: set by dispatch_* functions, drained by try_collect_pending.
    pending: Option<PendingReadback>,
}

impl GpuFimSolver {
    fn new(ctx: &GpuContext, k: usize, n: usize, width: usize, height: usize) -> Self {
        let device = &ctx.device;

        let tile_cols = (width + TILE - 1) / TILE;
        let tile_rows = (height + TILE - 1) / TILE;
        let num_tiles = tile_cols * tile_rows;

        let diag_cells = ((width * width + height * height) as f64).sqrt();
        let max_rounds = ((diag_cells / TILE as f64).ceil() as u32 * 2).max(8) + 4;
        let active_cap = (5 * k * num_tiles) as u32;

        // ── Params ────────────────────────────────────────────────────────────
        let params = GpuParams {
            w: width as u32,
            h: height as u32,
            k: k as u32,
            num_tile_cols: tile_cols as u32,
            num_tile_rows: tile_rows as u32,
            num_tiles: num_tiles as u32,
            n: n as u32,
            cell_size: 0.2_f32,
            conv_tol: CONV_TOL,
            active_cap,
            is_warm: 0,
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: unsafe { as_bytes(&params) },
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Data buffers ──────────────────────────────────────────────────────
        let u_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (k * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | if ctx.mappable {
                    wgpu::BufferUsages::MAP_READ
                } else {
                    wgpu::BufferUsages::empty()
                },
            mapped_at_creation: false,
        });
        let speed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (k * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_round_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (k * num_tiles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let make_active = |sz: u32| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (sz as usize * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let active_a = make_active(active_cap);
        let active_b = make_active(active_cap);

        let make_count = || {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &0u32.to_ne_bytes(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let count_a = make_count();
        let count_b = make_count();

        let indirect_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u32, 1u32, 1u32]
                .iter()
                .flat_map(|u| u.to_ne_bytes())
                .collect::<Vec<_>>(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        // True number of active_in entries for the current round.  The indirect
        // dispatch grid is rounded up to 2-D, so fim_update needs the exact count
        // to discard the overshoot workgroups.
        let threshold_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threshold_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One threshold per destination.
        let thresholds_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thresholds"),
            size: (k.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let round_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("round_count"),
            contents: &0u32.to_ne_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        // ── 4-byte convergence check buffer ───────────────────────────────────
        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Readback staging buffer, only when u_buf itself is not mappable ───
        let staging = (!ctx.mappable).then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (k * n * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // ── Bind groups ───────────────────────────────────────────────────────
        let fill_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.fill_bgl_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let bg_persistent = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.fim_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: speed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sources_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tile_round_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: round_count_buf.as_entire_binding(),
                },
            ],
        });
        let bg_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.fim_bgl_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: active_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: active_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_b.as_entire_binding(),
                },
            ],
        });
        let bg_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.fim_bgl_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: active_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: active_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_a.as_entire_binding(),
                },
            ],
        });
        let setup_bg_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.setup_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indirect_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: count_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: round_count_buf.as_entire_binding(),
                },
            ],
        });
        let setup_bg_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.setup_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indirect_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: count_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: round_count_buf.as_entire_binding(),
                },
            ],
        });

        GpuFimSolver {
            speed_scratch: Vec::with_capacity(n),
            u_buf,
            speed_buf,
            sources_buf,
            tile_round_buf,
            active_a,
            active_b,
            count_a,
            count_b,
            indirect_buf,
            round_count_buf,
            threshold_buf,
            thresholds_buf,
            count_staging,
            params_buf,
            staging,
            bg_persistent,
            bg_ping,
            bg_pong,
            setup_bg_ping,
            setup_bg_pong,
            fill_bg,
            k,
            n,
            width,
            height,
            tile_cols,
            tile_rows,
            num_tiles,
            max_rounds,
            active_cap,
            u_buf_valid: false,
            prev_sources_flat: Vec::new(),
            prev_src_offsets: Vec::new(),
            pending: None,
        }
    }

    // ── Per-solve uploads ─────────────────────────────────────────────────────

    fn upload_speed_and_sources(
        &mut self,
        ctx: &GpuContext,
        speed_field: &[f64],
        sources: &[u32],
        cell_size: f64,
        is_warm: u32,
    ) {
        self.speed_scratch.clear();
        self.speed_scratch
            .extend(speed_field.iter().map(|&v| v as f32));
        ctx.queue
            .write_buffer(&self.speed_buf, 0, f32_as_bytes(&self.speed_scratch));
        ctx.queue
            .write_buffer(&self.sources_buf, 0, u32_as_bytes(sources));

        let params = GpuParams {
            w: self.width as u32,
            h: self.height as u32,
            k: self.k as u32,
            num_tile_cols: self.tile_cols as u32,
            num_tile_rows: self.tile_rows as u32,
            num_tiles: self.num_tiles as u32,
            n: self.n as u32,
            cell_size: cell_size as f32,
            conv_tol: CONV_TOL,
            active_cap: self.active_cap,
            is_warm,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, unsafe { as_bytes(&params) });
    }

    /// f32-native variant: uploads `speed_field` directly without allocating a conversion buffer.
    fn upload_speed_and_sources_f32(
        &self,
        ctx: &GpuContext,
        speed_field: &[f32],
        sources: &[u32],
        cell_size: f32,
        is_warm: u32,
    ) {
        ctx.queue
            .write_buffer(&self.speed_buf, 0, f32_as_bytes(speed_field));
        ctx.queue
            .write_buffer(&self.sources_buf, 0, u32_as_bytes(sources));
        let params = GpuParams {
            w: self.width as u32,
            h: self.height as u32,
            k: self.k as u32,
            num_tile_cols: self.tile_cols as u32,
            num_tile_rows: self.tile_rows as u32,
            num_tiles: self.num_tiles as u32,
            n: self.n as u32,
            cell_size,
            conv_tol: CONV_TOL,
            active_cap: self.active_cap,
            is_warm,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, unsafe { as_bytes(&params) });
    }

    /// Reset every cell whose travel time is >= `threshold` back to infinity.
    ///
    /// This is what makes a warm restart correct.  FIM's update rule only ever
    /// *lowers* a value, so when a cell's cost rises the stale smaller value from
    /// the prior is a fixed point the solver can never correct upward.  Cells below
    /// the threshold are provably unaffected: the optimal path to such a cell is
    /// made entirely of cells with travel time <= its own, hence all < threshold,
    /// so no changed cell (every one of which has prior >= threshold) can lie on it.
    /// Everything at or above the threshold might route through a changed cell, so
    /// it is discarded and recomputed from the surviving inner region.
    fn gpu_threshold_reset(&self, ctx: &GpuContext, thresholds: &[f32]) {
        debug_assert_eq!(thresholds.len(), self.k);
        if thresholds.iter().all(|t| !t.is_finite() || *t <= 0.0) {
            if breakdown() {
                println!("[warm] no usable threshold -> NO reset performed");
            }
            return;
        }
        if breakdown() {
            let fin: Vec<f32> = thresholds.iter().copied().filter(|t| t.is_finite()).collect();
            println!(
                "[warm] per-dest thresholds: k={} min={:.1} max={:.1}",
                thresholds.len(),
                fin.iter().cloned().fold(f32::INFINITY, f32::min),
                fin.iter().cloned().fold(0.0f32, f32::max),
            );
        }
        // A non-finite threshold means "nothing changed for this destination";
        // +inf keeps every cell, which is what we want.
        let raw: Vec<u32> = thresholds.iter().map(|t| t.to_bits()).collect();
        ctx.queue
            .write_buffer(&self.thresholds_buf, 0, u32_as_bytes(&raw));

        let total = (self.k * self.n) as u32;
        let (bx, by) = split_dispatch(total.div_ceil(256));
        // ThresholdParams: total_elems, n, stride_x, _pad
        let params: [u32; 4] = [total, self.n as u32, bx * 256, 0];
        ctx.queue
            .write_buffer(&self.threshold_buf, 0, u32_as_bytes(&params));
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.threshold_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.threshold_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.thresholds_buf.as_entire_binding() },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.threshold_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(bx, by, 1);
        }
        ctx.queue.submit([enc.finish()]);
    }

    /// Append every tile holding a discarded cell adjacent to a surviving one, then
    /// republish the resulting seed count. Must run AFTER gpu_threshold_reset.
    ///
    /// `arm_seed_warm` left the CPU seed count in count_a; this pass appends to the
    /// same buffer, and the setup kernel then publishes the total into indirect_buf
    /// and round_count (and clears count_a for the ping-pong to reuse).
    fn gpu_seed_boundary(&self, ctx: &GpuContext) {
        let total = (self.k * self.n) as u32;
        let (bx, by) = split_dispatch(total.div_ceil(256));
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.boundary_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tile_round_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.active_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.count_a.as_entire_binding() },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.boundary_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(bx, by, 1);
        }
        // setup_bg_pong reads count_a: publishes the total seed count into
        // indirect_buf + round_count and resets count_a to 0.
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.setup_pipeline);
            pass.set_bind_group(0, &self.setup_bg_pong, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
    }

    /// Warm variant of arm_seed: leaves the seed count in count_a so the boundary
    /// pass can append to it, and does not publish indirect_buf/round_count -- the
    /// setup kernel at the end of gpu_seed_boundary does that instead.
    fn arm_seed_warm(&self, ctx: &GpuContext, slots: &[u32]) {
        ctx.queue.write_buffer(&self.active_a, 0, u32_as_bytes(slots));
        ctx.queue
            .write_buffer(&self.count_a, 0, &(slots.len() as u32).to_ne_bytes());
        ctx.queue
            .write_buffer(&self.count_b, 0, &0u32.to_ne_bytes());
        for &slot in slots {
            ctx.queue.write_buffer(
                &self.tile_round_buf,
                (slot as usize * 4) as u64,
                &1u32.to_ne_bytes(),
            );
        }
    }

    /// Reset every cell in the listed tile slots to ∞ on the GPU.
    fn gpu_reset_dirty_tiles(&self, ctx: &GpuContext, num_slots: u32) {
        if num_slots == 0 {
            return;
        }
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.reset_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.active_a.as_entire_binding(),
                },
                // `num_slots` always equals the count arm_seed just published here.
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.round_count_buf.as_entire_binding(),
                },
            ],
        });
        let (bx, by) = split_dispatch(num_slots);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.reset_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(bx, by, 1);
        }
        ctx.queue.submit([enc.finish()]);
    }

    // ── Readback plumbing ─────────────────────────────────────────────────────

    /// Buffer the CPU maps to read results: `u_buf` in place when the adapter
    /// allows it, otherwise the staging copy.
    fn readback_src(&self) -> &wgpu::Buffer {
        self.staging.as_ref().unwrap_or(&self.u_buf)
    }

    /// Blit into staging (only when a staging buffer exists) and request the map.
    /// Returns the submission to wait on, or `None` when reading `u_buf` in place
    /// because nothing new was submitted.
    fn issue_map(&self, ctx: &GpuContext) -> Option<wgpu::SubmissionIndex> {
        let si = self.staging.as_ref().map(|staging| {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.u_buf, 0, staging, 0, (self.k * self.n * 4) as u64);
            ctx.queue.submit([enc.finish()])
        });
        self.readback_src()
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        si
    }

    fn await_map(ctx: &GpuContext, si: Option<wgpu::SubmissionIndex>) {
        let _ = match si {
            Some(si) => ctx.device.poll(wgpu::PollType::Wait {
                submission_index: Some(si),
                timeout: None,
            }),
            // No submission of our own: the map can only fire once the FIM work
            // already queued has retired, so wait on everything outstanding.
            None => ctx.device.poll(wgpu::PollType::wait_indefinitely()),
        };
    }

    /// A mapped `u_buf` is unusable by the GPU, so every solve must start from an
    /// unmapped state. Callers that skipped `try_collect_pending` are drained here.
    fn ensure_readback_drained(&mut self, ctx: &GpuContext) {
        if self.pending.is_some() {
            self.collect_pending(ctx);
        }
    }

    fn readback_f32(
        &self,
        ctx: &GpuContext,
        out_ptrs: &[usize],
        gpu_start: Option<std::time::Instant>,
    ) {
        use rayon::prelude::*;

        let si = self.issue_map(ctx);
        Self::await_map(ctx, si);

        if let (Some(t), true) = (gpu_start, crate::PRINT_TIMINGS) {
            println!(
                "[GPU wgpu] GPU blit: {:.1}ms  ({:.0} MB)",
                t.elapsed().as_secs_f64() * 1e3,
                (self.k * self.n * 4) as f64 / 1e6,
            );
        }

        let t_memcpy = crate::PRINT_TIMINGS.then(std::time::Instant::now);

        let mapped = self.readback_src().slice(..).get_mapped_range();
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(mapped.as_ptr() as *const f32, self.k * self.n) };

        let n = self.n;
        src.par_chunks(n)
            .zip(out_ptrs.par_iter())
            .for_each(|(src_dest, &dst_ptr)| {
                let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f32, n) };
                dst.copy_from_slice(src_dest);
            });

        drop(mapped);
        self.readback_src().unmap();

        if let (Some(t), true) = (t_memcpy, crate::PRINT_TIMINGS) {
            println!(
                "[GPU wgpu] readback memcpy: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3,
            );
        }
    }

    // ── Deferred readback helpers ─────────────────────────────────────────────

    fn begin_readback_f32(&mut self, ctx: &GpuContext, out_ptrs: Vec<usize>) {
        let si = self.issue_map(ctx);
        self.pending = Some(PendingReadback {
            submission: si,
            out_ptrs,
            is_f32: true,
        });
    }

    fn begin_readback(&mut self, ctx: &GpuContext, out_ptrs: Vec<usize>) {
        let si = self.issue_map(ctx);
        self.pending = Some(PendingReadback {
            submission: si,
            out_ptrs,
            is_f32: false,
        });
    }

    fn collect_pending(&mut self, ctx: &GpuContext) {
        if let Some(p) = self.pending.take() {
            let t_total = crate::PRINT_TIMINGS.then(std::time::Instant::now);
            Self::await_map(ctx, p.submission);
            let t_after_poll = crate::PRINT_TIMINGS.then(std::time::Instant::now);
            if p.is_f32 {
                self.do_collect_f32(&p.out_ptrs);
            } else {
                self.do_collect_f64(&p.out_ptrs);
            }
            if let Some(t) = t_total {
                let total_ms = t.elapsed().as_secs_f64() * 1e3;
                let copy_ms = t_after_poll
                    .map(|t2| t2.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                println!(
                    "[GPU wgpu] collect_pending: poll {:.1}ms  copy {:.1}ms  total {:.1}ms",
                    total_ms - copy_ms,
                    copy_ms,
                    total_ms,
                );
            }
        }
    }

    fn do_collect_f32(&self, out_ptrs: &[usize]) {
        use rayon::prelude::*;
        let mapped = self.readback_src().slice(..).get_mapped_range();
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(mapped.as_ptr() as *const f32, self.k * self.n) };
        let n = self.n;
        src.par_chunks(n)
            .zip(out_ptrs.par_iter())
            .for_each(|(src_dest, &dst_ptr)| {
                let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f32, n) };
                dst.copy_from_slice(src_dest);
            });
        drop(mapped);
        self.readback_src().unmap();
    }

    fn do_collect_f64(&self, out_ptrs: &[usize]) {
        use rayon::prelude::*;
        const CHUNK: usize = 1 << 18;
        let mapped = self.readback_src().slice(..).get_mapped_range();
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(mapped.as_ptr() as *const f32, self.k * self.n) };
        let n = self.n;
        src.par_chunks(n)
            .zip(out_ptrs.par_iter())
            .for_each(|(src_dest, &dst_ptr)| {
                let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f64, n) };
                dst.par_chunks_mut(CHUNK)
                    .zip(src_dest.par_chunks(CHUNK))
                    .for_each(|(d, s)| {
                        for (&v, out) in s.iter().zip(d.iter_mut()) {
                            *out = v as f64;
                        }
                    });
            });
        drop(mapped);
        self.readback_src().unmap();
    }

    fn reset_tile_round(&self, ctx: &GpuContext) {
        let zeros = vec![0u8; self.k * self.num_tiles * 4];
        ctx.queue.write_buffer(&self.tile_round_buf, 0, &zeros);
    }

    fn gpu_fill_sentinel(&self, ctx: &GpuContext) {
        let total_elems = (self.k * self.n) as u32;
        let total_wg = total_elems.div_ceil(256);
        let bx = total_wg.min(65535);
        let by = total_wg.div_ceil(bx);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.fill_pipeline);
            pass.set_bind_group(0, &self.fill_bg, &[]);
            pass.dispatch_workgroups(bx, by, 1);
        }
        ctx.queue.submit([enc.finish()]);
    }

    fn write_source_cells(&self, ctx: &GpuContext, d: usize, srcs: &[u32]) {
        for &src in srcs {
            let off = ((d * self.n + src as usize) * 4) as u64;
            ctx.queue
                .write_buffer(&self.u_buf, off, &0.0f32.to_ne_bytes());
        }
    }

    fn init_u_cold_ms(&self, ctx: &GpuContext, sources_flat: &[u32], src_offsets: &[u32]) {
        self.gpu_fill_sentinel(ctx);
        for d in 0..self.k {
            let srcs = &sources_flat[src_offsets[d] as usize..src_offsets[d + 1] as usize];
            self.write_source_cells(ctx, d, srcs);
        }
    }

    fn set_source_cells_ms(&self, ctx: &GpuContext, sources_flat: &[u32], src_offsets: &[u32]) {
        for d in 0..self.k {
            let srcs = &sources_flat[src_offsets[d] as usize..src_offsets[d + 1] as usize];
            self.write_source_cells(ctx, d, srcs);
        }
    }

    // ── Seed-tile computation ─────────────────────────────────────────────────

    fn seed_tiles_for_dest(&self, d: usize, srcs: &[u32], out: &mut HashSet<u32>) {
        for &src in srcs {
            let tile_r = (src as usize / self.width) / TILE;
            let tile_c = (src as usize % self.width) / TILE;
            self.add_tile_and_neighbors(d, tile_r, tile_c, out);
        }
    }

    fn seed_slots_cold_ms(&self, sources_flat: &[u32], src_offsets: &[u32]) -> Vec<u32> {
        let mut slots = HashSet::new();
        for d in 0..self.k {
            let srcs = &sources_flat[src_offsets[d] as usize..src_offsets[d + 1] as usize];
            self.seed_tiles_for_dest(d, srcs, &mut slots);
        }
        slots.into_iter().collect()
    }

    fn seed_slots_warm_ms(
        &self,
        changed_cells: &[u32],
        sources_flat: &[u32],
        src_offsets: &[u32],
    ) -> Vec<u32> {
        let mut slots = HashSet::new();
        for d in 0..self.k {
            let srcs = &sources_flat[src_offsets[d] as usize..src_offsets[d + 1] as usize];
            self.seed_tiles_for_dest(d, srcs, &mut slots);
            for &cell in changed_cells {
                let tile_r = (cell as usize / self.width) / TILE;
                let tile_c = (cell as usize % self.width) / TILE;
                self.add_tile_and_neighbors(d, tile_r, tile_c, &mut slots);
            }
        }
        slots.into_iter().collect()
    }

    fn add_tile_and_neighbors(
        &self,
        dest: usize,
        tile_r: usize,
        tile_c: usize,
        out: &mut HashSet<u32>,
    ) {
        let base = dest * self.num_tiles;
        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                if dr != 0 && dc != 0 {
                    continue;
                }
                let r = tile_r as i64 + dr;
                let c = tile_c as i64 + dc;
                if r >= 0 && r < self.tile_rows as i64 && c >= 0 && c < self.tile_cols as i64 {
                    out.insert((base + r as usize * self.tile_cols + c as usize) as u32);
                }
            }
        }
    }

    fn arm_seed(&self, ctx: &GpuContext, slots: &[u32]) {
        let cnt = slots.len() as u32;
        ctx.queue
            .write_buffer(&self.active_a, 0, u32_as_bytes(slots));
        ctx.queue
            .write_buffer(&self.count_a, 0, &0u32.to_ne_bytes());
        ctx.queue
            .write_buffer(&self.count_b, 0, &0u32.to_ne_bytes());
        // Same 2-D split as setup_indirect: the seed count also scales with k and
        // must not overflow a single dispatch dimension.
        let indirect_init: Vec<u8> = [cnt.min(MAX_WG_PER_DIM), cnt.div_ceil(MAX_WG_PER_DIM), 1u32]
            .iter()
            .flat_map(|u| u.to_ne_bytes())
            .collect();
        ctx.queue
            .write_buffer(&self.indirect_buf, 0, &indirect_init);
        ctx.queue
            .write_buffer(&self.round_count_buf, 0, &cnt.to_ne_bytes());
        for &slot in slots {
            let byte_offset = (slot as usize * 4) as u64;
            ctx.queue
                .write_buffer(&self.tile_round_buf, byte_offset, &1u32.to_ne_bytes());
        }
    }

    // ── Encode and submit ─────────────────────────────────────────────────────

    fn encode_and_submit(&self, ctx: &GpuContext) {
        // Rounds encoded per submit+poll cycle. Env-overridable to measure how the
        // blocking poll scales: changing it varies the number of poll() calls
        // without changing the total pass or dispatch count.
        let rounds_per_batch: u32 = {
            static V: OnceLock<u32> = OnceLock::new();
            *V.get_or_init(|| {
                std::env::var("EIKONAL_RPB")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(64)
            })
        };

        let (mut acc_active, mut acc_rounds) = (0u64, 0u64);
        let mut last_active = 0u32;
        let (mut acc_enc, mut acc_sub, mut acc_poll, mut acc_map) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut batches = 0u32;
        let mut t_phase = std::time::Instant::now();

        let mut round = 0u32;
        while round < self.max_rounds {
            let batch_start = round;
            let batch_end = (round + rounds_per_batch).min(self.max_rounds);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // One compute pass per dispatch is deliberate, despite the cost.
            //
            // wgpu-core closes the underlying command encoder at the end of every
            // compute pass and opens a fresh one to carry the barriers
            // ("(wgpu internal) Pre Pass"); wgpu-hal's Metal `begin_encoding` maps
            // that onto a new MTLCommandBuffer.  So this loop costs two Metal
            // command buffers per dispatch regardless of ROUNDS_PER_BATCH — the
            // batching only reduces `queue.submit` calls, not command buffers.
            //
            // Collapsing all rounds into a single compute pass removes ~64x of that
            // overhead and measurably speeds up the FIM phase, but it is NOT safe:
            // wgpu inserts no barrier between dispatches within a pass, and the
            // setup_indirect -> dispatch_workgroups_indirect handoff then races.
            // Measured at k=4 on 4107x3769, three runs each: pass-per-dispatch gave
            // 100% reachable cells every time; single-pass gave 100%, 41%, 100%.
            // Reduce dispatch *count* (fold setup_indirect into fim_update, or use
            // larger tiles) rather than removing the barriers between them.
            //
            // Folding setup_indirect is NOT worth it, measured rather than guessed:
            // injecting N empty compute passes per round on the EdSheeran arena
            // scenario (3 reps, N in {0,4,8,16}, k=10..16) gives a marginal cost of
            // 0.48 ms per destination per pass-per-round -- 11x the residual noise,
            // so well resolved. Folding removes one of two passes: 6.4 ms of a
            // ~299 ms solve, i.e. 2.1%, in exchange for a last-workgroup-wins
            // device-wide sync. Per-solve cost is dominated by CPU-side setup, not
            // by pass encoding: ~18 ms GPU compute + ~8 ms blit + ~13 ms encoding
            // leaves ~260 ms unaccounted (Metal System Trace, 3.0% GPU utilisation).
            while round < batch_end {
                let (fim_bg1, setup_bg) = if round % 2 == 0 {
                    (&self.bg_ping, &self.setup_bg_ping)
                } else {
                    (&self.bg_pong, &self.setup_bg_pong)
                };
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&ctx.fim_pipeline);
                    pass.set_bind_group(0, &self.bg_persistent, &[]);
                    pass.set_bind_group(1, fim_bg1, &[]);
                    pass.dispatch_workgroups_indirect(&self.indirect_buf, 0);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&ctx.setup_pipeline);
                    pass.set_bind_group(0, setup_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                round += 1;
            }

            // Read the convergence count from round_count, not indirect_buf[0]:
            // the latter is min(count, MAX_WG_PER_DIM) and so under-reports (it
            // pins at 65535) once the frontier is large. Both are 0 exactly when
            // the solve has converged, but round_count is the honest number.
            enc.copy_buffer_to_buffer(&self.round_count_buf, 0, &self.count_staging, 0, 4);
            let t_enc = t_phase.elapsed().as_secs_f64() * 1e3;
            let t_sub = std::time::Instant::now();
            let si = ctx.queue.submit([enc.finish()]);
            let ms_submit = t_sub.elapsed().as_secs_f64() * 1e3;
            let t_poll = std::time::Instant::now();

            self.count_staging
                .slice(..4)
                .map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::Wait {
                submission_index: Some(si),
                timeout: None,
            });
            let ms_poll = t_poll.elapsed().as_secs_f64() * 1e3;
            let t_map = std::time::Instant::now();
            let active = {
                let m = self.count_staging.slice(..4).get_mapped_range();
                u32::from_ne_bytes(m[0..4].try_into().unwrap())
            };
            self.count_staging.unmap();
            if breakdown() {
                acc_enc += t_enc;
                acc_sub += ms_submit;
                acc_poll += ms_poll;
                acc_map += t_map.elapsed().as_secs_f64() * 1e3;
                batches += 1;
            }
            t_phase = std::time::Instant::now();

            if crate::PRINT_TIMINGS {
                println!(
                    "[GPU wgpu] batch rounds {}-{}: active_next={}",
                    batch_start,
                    round,
                    active
                );
            }
            // Total tile activations. Exact only at EIKONAL_RPB=1, where each batch
            // is one round; at larger batch sizes this samples every RPB-th round.
            acc_active += active as u64;
            acc_rounds += 1u64;

            if active == 0 {
                break;
            }
            last_active = active;
        }
        // The loop can also end by exhausting max_rounds with tiles still queued, in
        // which case the field is silently truncated -- some cells keep whatever
        // partial value they had. max_rounds is derived from diag/TILE, i.e. it
        // assumes the frontier advances one tile per round, which tortuous geometry
        // violates (the path length is far longer than the diagonal). It also scales
        // as 1/TILE, so a larger TILE truncates sooner and would look spuriously fast
        // in a TILE comparison. Always report it rather than let it pass unnoticed.
        if round >= self.max_rounds && last_active != 0 {
            eprintln!(
                "[eikonal] WARNING: hit max_rounds={} with {last_active} tiles still \
active -- travel-time field is TRUNCATED and some cells are too large. \
TILE={TILE}, grid {}x{}.",
                self.max_rounds, self.width, self.height
            );
        }
        if breakdown() {
            println!(
                "[encode] batches={batches} encode={acc_enc:.1} submit={acc_sub:.1} \
poll={acc_poll:.1} mapread={acc_map:.1} activations={acc_active} sampled_rounds={acc_rounds} \
rounds={round}/{}",
                self.max_rounds
            );
        }
    }

    // ── Readback ──────────────────────────────────────────────────────────────

    fn readback(
        &self,
        ctx: &GpuContext,
        out_ptrs: &[usize],
        gpu_start: Option<std::time::Instant>,
    ) {
        use rayon::prelude::*;

        let si = self.issue_map(ctx);
        Self::await_map(ctx, si);

        if let (Some(t), true) = (gpu_start, crate::PRINT_TIMINGS) {
            println!(
                "[GPU wgpu] GPU blit: {:.1}ms  ({:.0} MB)",
                t.elapsed().as_secs_f64() * 1e3,
                (self.k * self.n * 4) as f64 / 1e6,
            );
        }

        let t_convert = crate::PRINT_TIMINGS.then(std::time::Instant::now);

        let mapped = self.readback_src().slice(..).get_mapped_range();
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(mapped.as_ptr() as *const f32, self.k * self.n) };

        const CHUNK: usize = 1 << 18;
        let n = self.n;
        src.par_chunks(n)
            .zip(out_ptrs.par_iter())
            .for_each(|(src_dest, &dst_ptr)| {
                let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f64, n) };
                dst.par_chunks_mut(CHUNK)
                    .zip(src_dest.par_chunks(CHUNK))
                    .for_each(|(d, s)| {
                        for (&v, out) in s.iter().zip(d.iter_mut()) {
                            *out = v as f64;
                        }
                    });
            });

        drop(mapped);
        self.readback_src().unmap();

        if let (Some(t), true) = (t_convert, crate::PRINT_TIMINGS) {
            println!(
                "[GPU wgpu] readback convert: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3,
            );
        }
    }
}

// ── Bind-group-layout helpers ─────────────────────────────────────────────────

fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn storage_rw() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn storage_r() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

// ── Singletons ────────────────────────────────────────────────────────────────

static MAX_BINDING: OnceLock<u64> = OnceLock::new();

fn adapter_max_binding() -> u64 {
    *MAX_BINDING.get_or_init(|| {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map(|a| a.limits().max_storage_buffer_binding_size as u64)
        .unwrap_or(128 * 1024 * 1024)
    })
}

struct GpuState {
    ctx: GpuContext,
    solver: Option<GpuFimSolver>,
    width: usize,
    height: usize,
}

static GPU_STATE: OnceLock<Mutex<Option<GpuState>>> = OnceLock::new();

pub fn fits_in_gpu(k: usize, n: usize) -> bool {
    let per_buf = k.saturating_mul(n).saturating_mul(4) as u64;
    per_buf <= adapter_max_binding() * 4 / 5
}

fn with_solver<F, R>(k: usize, n: usize, width: usize, height: usize, f: F) -> R
where
    F: FnOnce(&GpuContext, &mut GpuFimSolver) -> R,
{
    let lock = GPU_STATE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap();

    let grid_changed = guard
        .as_ref()
        .map_or(true, |s| s.width != width || s.height != height);

    if grid_changed {
        *guard = None;
        let t_dev = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        let ctx = GpuContext::new();
        if let Some(t) = t_dev {
            println!(
                "[GPU wgpu] device+pipeline init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_buf = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        let solver = GpuFimSolver::new(&ctx, k, n, width, height);
        if let Some(t) = t_buf {
            println!(
                "[GPU wgpu] solver buffers alloc: {:.1}ms  (k={k} n={n} grid={width}×{height})",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        *guard = Some(GpuState {
            ctx,
            solver: Some(solver),
            width,
            height,
        });
    } else {
        let state = guard.as_mut().unwrap();
        let solver_changed = state.solver.as_ref().map_or(true, |s| s.k != k || s.n != n);
        if solver_changed {
            state.solver = None;
            let _ = state.ctx.device.poll(wgpu::PollType::wait_indefinitely());
            let t_buf = crate::PRINT_TIMINGS.then(std::time::Instant::now);
            state.solver = Some(GpuFimSolver::new(&state.ctx, k, n, width, height));
            if let Some(t) = t_buf {
                println!(
                    "[GPU wgpu] solver buffers alloc: {:.1}ms  (k={k} n={n} grid={width}×{height})",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }

    let state = guard.as_mut().unwrap();
    let ctx = &state.ctx;
    let solver = state.solver.as_mut().unwrap();
    // When u_buf is mapped in place it cannot be touched by the GPU, so a caller
    // that dispatched without collecting would otherwise fault the next solve.
    // Draining here also gives that caller its results rather than losing them.
    solver.ensure_readback_drained(ctx);
    f(ctx, solver)
}

// ── Public entry points ───────────────────────────────────────────────────────

pub fn solve_cold(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let src_offsets: Vec<u32> = (0..=out_ptrs.len() as u32).collect();
    solve_cold_ms(
        out_ptrs,
        n,
        speed_field,
        sources,
        &src_offsets,
        width,
        height,
        cell_size,
    );
}

pub fn solve_warm(
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
    let src_offsets: Vec<u32> = (0..=out_ptrs.len() as u32).collect();
    solve_warm_ms(
        out_ptrs,
        prior_ptrs,
        n,
        speed_field,
        changed_cells,
        sources,
        &src_offsets,
        width,
        height,
        cell_size,
    );
}

/// Smallest prior travel time over the changed cells, across every destination.
///
/// This is the threshold that separates the provably-safe inner region (kept) from
/// the region that may route through a changed cell (reset). Taking the minimum
/// across all destinations is conservative -- it resets at least as much as any
/// per-destination threshold would -- which is safe but does more work when the
/// destinations differ widely.
fn min_prior_over_changed_f32(prior_ptrs: &[usize], n: usize, changed: &[u32]) -> Vec<f32> {
    prior_ptrs
        .iter()
        .map(|&raw| {
            let prior = unsafe { std::slice::from_raw_parts(raw as *const f32, n) };
            changed
                .iter()
                .map(|&c| prior[c as usize])
                .fold(f32::INFINITY, f32::min)
        })
        .collect()
}

/// f64 prior variant of `min_prior_over_changed_f32`.
fn min_prior_over_changed_f64(prior_ptrs: &[usize], n: usize, changed: &[u32]) -> Vec<f32> {
    prior_ptrs
        .iter()
        .map(|&raw| {
            let prior = unsafe { std::slice::from_raw_parts(raw as *const f64, n) };
            changed
                .iter()
                .map(|&c| prior[c as usize])
                .fold(f64::INFINITY, f64::min) as f32
        })
        .collect()
}

fn first_sources(sources_flat: &[u32], src_offsets: &[u32]) -> Vec<u32> {
    let k = src_offsets.len().saturating_sub(1);
    (0..k)
        .map(|d| sources_flat[src_offsets[d] as usize])
        .collect()
}

pub fn solve_cold_ms(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        s.upload_speed_and_sources(ctx, speed_field, &first, cell_size, 0);
        s.reset_tile_round(ctx);
        let slots = s.seed_slots_cold_ms(sources_flat, src_offsets);
        s.arm_seed(ctx, &slots);
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] upload+init (ms): {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        let t_blit = if crate::PRINT_TIMINGS {
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(t) = t_gpu {
                println!(
                    "[GPU wgpu] GPU FIM: {:.1}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            Some(std::time::Instant::now())
        } else {
            None
        };
        s.readback(ctx, out_ptrs, t_blit);
        s.u_buf_valid = true;
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
    });
}

pub fn solve_warm_ms(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    changed_cells: &[u32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        s.upload_speed_and_sources(ctx, speed_field, &first, cell_size, 1);
        let warm = s.u_buf_valid;
        if warm {
            let sources_changed =
                s.prev_sources_flat != sources_flat || s.prev_src_offsets != src_offsets;
            if sources_changed {
                for (d, &prior_raw) in prior_ptrs.iter().enumerate() {
                    let pf64 = unsafe { std::slice::from_raw_parts(prior_raw as *const f64, n) };
                    s.speed_scratch.clear();
                    s.speed_scratch.extend(pf64.iter().map(|&v| v as f32));
                    ctx.queue.write_buffer(
                        &s.u_buf,
                        (d * n * 4) as u64,
                        f32_as_bytes(&s.speed_scratch),
                    );
                }
            }
        } else {
            s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        }
        // Must be computed from the CPU-side prior before u_buf is disturbed.
        let u_threshold = if warm && !changed_cells.is_empty() {
            min_prior_over_changed_f64(prior_ptrs, n, changed_cells)
        } else {
            vec![f32::INFINITY; k]
        };
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
        s.reset_tile_round(ctx);
        let slots = if warm && !changed_cells.is_empty() {
            s.seed_slots_warm_ms(changed_cells, sources_flat, src_offsets)
        } else {
            s.seed_slots_cold_ms(sources_flat, src_offsets)
        };
        if warm {
            s.arm_seed_warm(ctx, &slots);
            // Threshold reset, not dirty-tile reset: resetting only the seeded tiles
            // leaves stale-small values everywhere else, which FIM cannot raise.
            s.gpu_threshold_reset(ctx, &u_threshold);
            s.set_source_cells_ms(ctx, sources_flat, src_offsets);
            s.gpu_seed_boundary(ctx);
        } else {
            s.arm_seed(ctx, &slots);
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        let t_blit = if crate::PRINT_TIMINGS {
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(t) = t_gpu {
                println!(
                    "[GPU wgpu] GPU FIM: {:.1}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            Some(std::time::Instant::now())
        } else {
            None
        };
        s.readback(ctx, out_ptrs, t_blit);
        s.u_buf_valid = true;
    });
}


// ── Per-solve phase breakdown ─────────────────────────────────────────────────
// EIKONAL_BREAKDOWN=1 prints one line per cold solve splitting the wall time into
// its phases. Kept in the tree because the interesting cost is CPU-side setup, not
// GPU work: a Metal System Trace of the EdSheeran scenario showed 3.0% GPU
// utilisation with ~18 ms of GPU compute inside a ~300 ms solve, so the question
// "where did the other 260 ms go" comes up repeatedly. Overhead is a handful of
// Instant::now() calls per solve, so this is cheap enough to leave enabled.
fn breakdown() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("EIKONAL_BREAKDOWN").is_ok())
}

pub fn solve_cold_ms_f32(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f32,
) {
    let k = out_ptrs.len();
    let bd = breakdown();
    let t_enter = std::time::Instant::now();
    with_solver(k, n, width, height, |ctx, s| {
        // Time inside the closure: with_solver may have rebuilt every buffer if k
        // changed, which at k=16 on a 15.5M-cell grid is ~1 GB of reallocation.
        let ms_solver = t_enter.elapsed().as_secs_f64() * 1e3;
        let mut t = std::time::Instant::now();
        let mut lap = |t: &mut std::time::Instant| {
            let v = t.elapsed().as_secs_f64() * 1e3;
            *t = std::time::Instant::now();
            v
        };

        let first = first_sources(sources_flat, src_offsets);
        let ms_first = lap(&mut t);
        s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        let ms_init = lap(&mut t);
        s.upload_speed_and_sources_f32(ctx, speed_field, &first, cell_size, 0);
        let ms_upload = lap(&mut t);
        s.reset_tile_round(ctx);
        let ms_reset = lap(&mut t);
        let slots = s.seed_slots_cold_ms(sources_flat, src_offsets);
        let ms_seed = lap(&mut t);
        s.arm_seed(ctx, &slots);
        let ms_arm = lap(&mut t);
        s.encode_and_submit(ctx);
        let ms_encode = lap(&mut t);
        s.readback_f32(ctx, out_ptrs, None);
        let ms_readback = lap(&mut t);
        s.u_buf_valid = true;
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
        let ms_bookkeep = lap(&mut t);

        if bd {
            let total = ms_solver + ms_first + ms_init + ms_upload + ms_reset
                + ms_seed + ms_arm + ms_encode + ms_readback + ms_bookkeep;
            println!(
                "[breakdown] k={k} slots={} | solver={ms_solver:.1} first={ms_first:.1} \
init={ms_init:.1} upload={ms_upload:.1} reset={ms_reset:.1} seed={ms_seed:.1} \
arm={ms_arm:.1} encode={ms_encode:.1} readback={ms_readback:.1} \
bookkeep={ms_bookkeep:.1} | total={total:.1}ms",
                slots.len()
            );
        }
    });
}

pub fn solve_warm_ms_f32(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed_field: &[f32],
    changed_cells: &[u32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f32,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        let first = first_sources(sources_flat, src_offsets);
        s.upload_speed_and_sources_f32(ctx, speed_field, &first, cell_size, 1);
        let warm = s.u_buf_valid;
        if warm {
            let sources_changed =
                s.prev_sources_flat != sources_flat || s.prev_src_offsets != src_offsets;
            if sources_changed {
                for (d, &prior_raw) in prior_ptrs.iter().enumerate() {
                    let prior = unsafe { std::slice::from_raw_parts(prior_raw as *const f32, n) };
                    ctx.queue
                        .write_buffer(&s.u_buf, (d * n * 4) as u64, f32_as_bytes(prior));
                }
            }
        } else {
            s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        }
        // Must be computed from the CPU-side prior before u_buf is disturbed.
        let u_threshold = if warm && !changed_cells.is_empty() {
            min_prior_over_changed_f32(prior_ptrs, n, changed_cells)
        } else {
            vec![f32::INFINITY; k]
        };
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
        s.reset_tile_round(ctx);
        let slots = if warm && !changed_cells.is_empty() {
            s.seed_slots_warm_ms(changed_cells, sources_flat, src_offsets)
        } else {
            s.seed_slots_cold_ms(sources_flat, src_offsets)
        };
        if warm {
            s.arm_seed_warm(ctx, &slots);
            // Threshold reset, not dirty-tile reset: resetting only the seeded tiles
            // leaves stale-small values everywhere else, which FIM cannot raise.
            s.gpu_threshold_reset(ctx, &u_threshold);
            s.set_source_cells_ms(ctx, sources_flat, src_offsets);
            s.gpu_seed_boundary(ctx);
        } else {
            s.arm_seed(ctx, &slots);
        }
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] upload+init (ms): {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3,
            );
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        let t_blit = if crate::PRINT_TIMINGS {
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(t) = t_gpu {
                println!(
                    "[GPU wgpu] GPU FIM: {:.1}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            Some(std::time::Instant::now())
        } else {
            None
        };
        s.readback_f32(ctx, out_ptrs, t_blit);
        s.u_buf_valid = true;
    });
}

// ── Deferred (non-blocking) dispatch entry points ─────────────────────────────

pub fn dispatch_warm_ms_f32(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed_field: &[f32],
    changed_cells: &[u32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f32,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.upload_speed_and_sources_f32(ctx, speed_field, &first, cell_size, 1);
        let warm = s.u_buf_valid;
        if warm {
            let sources_changed =
                s.prev_sources_flat != sources_flat || s.prev_src_offsets != src_offsets;
            if sources_changed {
                for (d, &prior_raw) in prior_ptrs.iter().enumerate() {
                    let prior = unsafe { std::slice::from_raw_parts(prior_raw as *const f32, n) };
                    ctx.queue
                        .write_buffer(&s.u_buf, (d * n * 4) as u64, f32_as_bytes(prior));
                }
            }
        } else {
            s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        }
        // Must be computed from the CPU-side prior before u_buf is disturbed.
        let u_threshold = if warm && !changed_cells.is_empty() {
            min_prior_over_changed_f32(prior_ptrs, n, changed_cells)
        } else {
            vec![f32::INFINITY; k]
        };
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
        s.reset_tile_round(ctx);
        let slots = if warm && !changed_cells.is_empty() {
            s.seed_slots_warm_ms(changed_cells, sources_flat, src_offsets)
        } else {
            s.seed_slots_cold_ms(sources_flat, src_offsets)
        };
        if warm {
            s.arm_seed_warm(ctx, &slots);
            s.gpu_threshold_reset(ctx, &u_threshold);
            s.set_source_cells_ms(ctx, sources_flat, src_offsets);
            s.gpu_seed_boundary(ctx);
        } else {
            s.arm_seed(ctx, &slots);
        }
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] dispatch upload+init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] dispatch GPU compute: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        s.begin_readback_f32(ctx, out_ptrs.to_vec());
        s.u_buf_valid = true;
    });
}

pub fn dispatch_cold_ms_f32(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f32,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        s.upload_speed_and_sources_f32(ctx, speed_field, &first, cell_size, 0);
        s.reset_tile_round(ctx);
        let slots = s.seed_slots_cold_ms(sources_flat, src_offsets);
        s.arm_seed(ctx, &slots);
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] dispatch upload+init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] dispatch GPU compute: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        s.begin_readback_f32(ctx, out_ptrs.to_vec());
        s.u_buf_valid = true;
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
    });
}

pub fn dispatch_warm_ms(
    out_ptrs: &[usize],
    prior_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    changed_cells: &[u32],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.upload_speed_and_sources(ctx, speed_field, &first, cell_size, 1);
        let warm = s.u_buf_valid;
        if warm {
            let sources_changed =
                s.prev_sources_flat != sources_flat || s.prev_src_offsets != src_offsets;
            if sources_changed {
                for (d, &prior_raw) in prior_ptrs.iter().enumerate() {
                    let pf64 = unsafe { std::slice::from_raw_parts(prior_raw as *const f64, n) };
                    s.speed_scratch.clear();
                    s.speed_scratch.extend(pf64.iter().map(|&v| v as f32));
                    ctx.queue.write_buffer(
                        &s.u_buf,
                        (d * n * 4) as u64,
                        f32_as_bytes(&s.speed_scratch),
                    );
                }
            }
        } else {
            s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        }
        // Must be computed from the CPU-side prior before u_buf is disturbed.
        let u_threshold = if warm && !changed_cells.is_empty() {
            min_prior_over_changed_f64(prior_ptrs, n, changed_cells)
        } else {
            vec![f32::INFINITY; k]
        };
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
        s.reset_tile_round(ctx);
        let slots = if warm && !changed_cells.is_empty() {
            s.seed_slots_warm_ms(changed_cells, sources_flat, src_offsets)
        } else {
            s.seed_slots_cold_ms(sources_flat, src_offsets)
        };
        if warm {
            s.arm_seed_warm(ctx, &slots);
            s.gpu_threshold_reset(ctx, &u_threshold);
            s.set_source_cells_ms(ctx, sources_flat, src_offsets);
            s.gpu_seed_boundary(ctx);
        } else {
            s.arm_seed(ctx, &slots);
        }
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] dispatch upload+init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] dispatch GPU compute: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        s.begin_readback(ctx, out_ptrs.to_vec());
        s.u_buf_valid = true;
    });
}

pub fn dispatch_cold_ms(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    sources_flat: &[u32],
    src_offsets: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        s.upload_speed_and_sources(ctx, speed_field, &first, cell_size, 0);
        s.reset_tile_round(ctx);
        let slots = s.seed_slots_cold_ms(sources_flat, src_offsets);
        s.arm_seed(ctx, &slots);
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] dispatch upload+init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] dispatch GPU compute: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        s.begin_readback(ctx, out_ptrs.to_vec());
        s.u_buf_valid = true;
        s.prev_sources_flat = sources_flat.to_vec();
        s.prev_src_offsets = src_offsets.to_vec();
    });
}

pub fn try_collect_pending(width: usize, height: usize) {
    let lock = GPU_STATE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap();
    if let Some(state) = guard.as_mut() {
        if state.width == width && state.height == height {
            let ctx = &state.ctx;
            if let Some(solver) = state.solver.as_mut() {
                solver.collect_pending(ctx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnoses the warm-restart failure mode: is the warm result too SMALL
    /// (monotone update rule cannot raise a stale value) or are cells UNREACHED
    /// (activation/dedup dropped work)? The two have opposite signatures.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_warm_failure_signature() {
        let (w, h) = (129usize, 129usize);
        let n = w * h;
        let src = ((h / 2) * w + h / 2) as u32;

        let speed_old = vec![1.0f32; n];
        let mut cold_old = vec![0.0f32; n];
        solve_cold_ms_f32(&[cold_old.as_mut_ptr() as usize], n, &speed_old, &[src], &[0, 1], w, h, 1.0);

        // Slow a patch down: true travel time must INCREASE downstream of it.
        let mut speed_new = speed_old.clone();
        for r in 20..40 {
            for c in 20..40 {
                speed_new[r * w + c] = 0.25;
            }
        }
        let changed: Vec<u32> = (0..n)
            .filter(|&i| (speed_new[i] - speed_old[i]).abs() > 1e-12)
            .map(|i| i as u32)
            .collect();

        let mut warm = vec![0.0f32; n];
        solve_warm_ms_f32(&[warm.as_mut_ptr() as usize], &[cold_old.as_ptr() as usize],
                          n, &speed_new, &changed, &[src], &[0, 1], w, h, 1.0);
        let mut cold_new = vec![0.0f32; n];
        solve_cold_ms_f32(&[cold_new.as_mut_ptr() as usize], n, &speed_new, &[src], &[0, 1], w, h, 1.0);

        let (mut too_small, mut too_big, mut unreached) = (0usize, 0usize, 0usize);
        let (mut worst_small, mut worst_big) = (0.0f32, 0.0f32);
        for i in 0..n {
            if !cold_new[i].is_finite() { continue; }
            if !warm[i].is_finite() { unreached += 1; continue; }
            let d = warm[i] - cold_new[i];
            if d < -0.05 { too_small += 1; worst_small = worst_small.min(d); }
            else if d > 0.05 { too_big += 1; worst_big = worst_big.max(d); }
        }
        println!("warm vs cold_new: too_small={too_small} (worst {worst_small:.3})  \
too_big={too_big} (worst {worst_big:.3})  unreached={unreached}");
        println!("  too_small  => monotone update rule cannot raise stale values");
        println!("  unreached  => activation / tile_round dedup dropped work");
    }

    /// Does a warm restart actually save time at production scale? The threshold
    /// reset keeps only cells below min(prior) over the changed set, so the payoff
    /// depends entirely on how far the change is from the source: a change close in
    /// resets nearly everything and degenerates to a cold solve.
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_warm_vs_cold() {
        let (w, h) = (4107usize, 3769usize);
        let n = w * h;
        let k: usize = 4;
        let speed = vec![1.0f32; n];
        let sources: Vec<u32> = (0..k)
            .map(|d| (((h / (k + 1)) * (d + 1)) * w + 20) as u32)
            .collect();
        let offs: Vec<u32> = (0..=k as u32).collect();

        let mut priors: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; n]).collect();
        let pp: Vec<usize> = priors.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
        let t = std::time::Instant::now();
        solve_cold_ms_f32(&pp, n, &speed, &sources, &offs, w, h, 0.2);
        let cold_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("\ncold solve (k={k}, {w}x{h}): {cold_ms:.0} ms\n");

        let prior_ptrs: Vec<usize> = priors.iter().map(|v| v.as_ptr() as usize).collect();
        for (label, cfrac) in [("change at 10% of width", 0.10f32),
                               ("change at 50% of width", 0.50),
                               ("change at 90% of width", 0.90)] {
            let c0 = (w as f32 * cfrac) as usize;
            let r0 = h / 2;
            let mut ns = speed.clone();
            let mut changed = Vec::new();
            for r in r0..(r0 + 40).min(h) {
                for c in c0..(c0 + 40).min(w) {
                    ns[r * w + c] = 0.3;
                    changed.push((r * w + c) as u32);
                }
            }
            let mut outs: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; n]).collect();
            let op: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
            let t = std::time::Instant::now();
            solve_warm_ms_f32(&op, &prior_ptrs, n, &ns, &changed, &sources, &offs, w, h, 0.2);
            let warm_ms = t.elapsed().as_secs_f64() * 1e3;
            println!("{label:24} warm {warm_ms:7.0} ms  vs cold {cold_ms:.0} ms  \
-> {:.2}x", cold_ms / warm_ms);
        }
    }

    // ── Bucketed tile scheduling prototype ────────────────────────────────────
    //
    // Mirrors the GPU kernel exactly so the counts transfer: TILE x TILE tiles,
    // a 1-cell halo snapshotted once per activation (the halo does NOT update
    // during the tile's inner iterations), TILE Gauss-Seidel sweeps over the
    // interior, monotone-decreasing updates, and directional neighbour activation.
    //
    // Two schedulers over identical machinery:
    //   All     - process every active tile each dispatch (what the GPU does now)
    //   Bucket  - process only tiles whose current min(u) falls in the lowest
    //             band [gmin, gmin + W), i.e. Kim's GroupMarching gate applied at
    //             tile granularity, using RUNNING values rather than converged ones.
    struct Proto<'a> {
        w: usize, h: usize, tc: usize, tr: usize,
        cs: f32, tol: f32,
        speed: &'a [f32],
        src: usize,
        u: Vec<f32>,
    }

    impl<'a> Proto<'a> {
        fn new(w: usize, h: usize, speed: &'a [f32], src: usize, cs: f32, tol: f32) -> Self {
            let mut u = vec![f32::INFINITY; w * h];
            u[src] = 0.0;
            Proto { w, h, tc: w.div_ceil(TILE), tr: h.div_ceil(TILE), cs, tol, speed, src, u }
        }
        #[inline]
        fn at(&self, r: i64, c: i64) -> f32 {
            if r < 0 || c < 0 || r >= self.h as i64 || c >= self.w as i64 { f32::INFINITY }
            else { self.u[r as usize * self.w + c as usize] }
        }
        /// Bucket key: earliest arrival time anywhere in the tile's snapshotted
        /// block, i.e. the min over interior AND the 1-cell halo.
        ///
        /// The halo is not optional. A tile activated by its neighbour has not been
        /// touched yet -- its interior is still all INFINITY -- so an interior-only
        /// key is INFINITY for every freshly queued tile. Once every tile in the
        /// active set is in that state, gmin is INFINITY, the cut is INFINITY, no
        /// tile satisfies key < cut, the batch comes back empty and the scheduler
        /// spins forever on a non-empty active set. Keying on the halo gives a
        /// finite key at activation: the time the wave actually reaches the tile.
        ///
        /// This is free on the GPU -- the workgroup already loads the halo block
        /// into shared memory, so the key is a reduction over data it holds.
        fn tile_key(&self, t: usize) -> f32 {
            let (tr0, tc0) = ((t / self.tc) as i64 * TILE as i64, (t % self.tc) as i64 * TILE as i64);
            let mut m = f32::INFINITY;
            for r in (tr0 - 1)..=(tr0 + TILE as i64) {
                for c in (tc0 - 1)..=(tc0 + TILE as i64) {
                    let v = self.at(r, c);
                    if v < m { m = v; }
                }
            }
            m
        }
        /// One tile activation. Returns edge flags: bit0 any change, 1 top, 2 bottom,
        /// 3 left, 4 right -- matching FLAG_* in fim_update.wgsl.
        fn process(&mut self, t: usize, snap: &[f32]) -> u32 {
            self.process_n(t, snap, TILE, false)
        }
        /// `iters` in-tile iterations. `gs` selects raster Gauss-Seidel (each cell
        /// consumes values written earlier in the same iteration) instead of the
        /// Jacobi model of the current kernel.
        fn process_n(&mut self, t: usize, snap: &[f32], iters: usize, gs: bool) -> u32 {
            let s = TILE + 2;
            let (tr0, tc0) = ((t / self.tc) * TILE, (t % self.tc) * TILE);
            // Load the block from the DISPATCH-START snapshot, not from live u.
            // Every workgroup in a GPU dispatch runs concurrently and loads its halo
            // from the state as it was when the dispatch began; reading live u instead
            // would make each tile see values written by tiles processed earlier in the
            // same batch -- Gauss-Seidel across tiles rather than Jacobi -- which
            // converges far faster than the hardware actually does and understates the
            // redundancy this prototype exists to measure. Writes stay on live u: tile
            // interiors are disjoint, so they never conflict.
            let at = |r: i64, c: i64| -> f32 {
                if r < 0 || c < 0 || r >= self.h as i64 || c >= self.w as i64 { f32::INFINITY }
                else { snap[r as usize * self.w + c as usize] }
            };
            let mut sm = vec![f32::INFINITY; s * s];
            for lr in 0..s {
                for lc in 0..s {
                    sm[lr * s + lc] = at(tr0 as i64 + lr as i64 - 1, tc0 as i64 + lc as i64 - 1);
                }
            }
            let mut flags = 0u32;
            // JACOBI, not Gauss-Seidel. Each iteration in fim_update.wgsl is one
            // update per thread separated by workgroupBarrier(), so every cell reads
            // its neighbours as they stood at the START of the iteration. Sweeping in
            // raster order over live `sm` instead would let a cell consume a value its
            // left/upper neighbour produced in the same iteration -- that converges a
            // tile in roughly one activation where the hardware needs many, and it is
            // what made this prototype report 3x tile redundancy against the GPU's 39x.
            let mut nx = sm.clone();
            for _ in 0..iters {
                for lr in 1..=TILE {
                    for lc in 1..=TILE {
                        let (r, c) = (tr0 + lr - 1, tc0 + lc - 1);
                        if r >= self.h || c >= self.w { continue; }
                        let x = r * self.w + c;
                        if x == self.src || self.speed[x] <= 0.0 { continue; }
                        let i = lr * s + lc;
                        let a = sm[i - 1].min(sm[i + 1]);
                        let b = sm[i - s].min(sm[i + s]);
                        let cost = self.cs / self.speed[x];
                        let (lo, hi) = (a.min(b), a.max(b));
                        let u1 = lo + cost;
                        let cand = if u1 <= hi { u1 } else {
                            let disc = 2.0 * cost * cost - (a - b) * (a - b);
                            if disc >= 0.0 { (a + b + disc.sqrt()) * 0.5 } else { u1 }
                        };
                        if cand < sm[i] - self.tol {
                            nx[i] = cand;
                            if gs { sm[i] = cand; }
                            flags |= 1;
                            if lr == 1 { flags |= 2; }
                            if lr == TILE { flags |= 4; }
                            if lc == 1 { flags |= 8; }
                            if lc == TILE { flags |= 16; }
                        }
                    }
                }
                if !gs { sm.copy_from_slice(&nx); }   // the barrier
            }
            for lr in 1..=TILE {
                for lc in 1..=TILE {
                    let (r, c) = (tr0 + lr - 1, tc0 + lc - 1);
                    if r < self.h && c < self.w { self.u[r * self.w + c] = sm[lr * s + lc]; }
                }
            }
            flags
        }
    }

    /// In-tile iterations, mirroring ITERS in fim_update.wgsl.
    const PROTO_ITERS: usize = 2 * TILE - 1;

    /// Returns (field, tile-processings, dispatches, per-dispatch batch sizes).
    fn proto_run(w: usize, h: usize, speed: &[f32], src: usize, cs: f32, band: Option<f32>)
        -> (Vec<f32>, usize, usize, Vec<usize>)
    { proto_run_n(w, h, speed, src, cs, band, PROTO_ITERS, false) }

    fn proto_run_n(w: usize, h: usize, speed: &[f32], src: usize, cs: f32, band: Option<f32>,
                   iters: usize, gs: bool)
        -> (Vec<f32>, usize, usize, Vec<usize>)
    {
        let mut sizes: Vec<usize> = Vec::new();
        let mut p = Proto::new(w, h, speed, src, cs, 1e-2);
        let nt = p.tc * p.tr;
        let mut inset = vec![false; nt];
        let mut key = vec![f32::INFINITY; nt];
        let mut act: Vec<usize> = Vec::new();
        let seed = (src / w / TILE) * p.tc + (src % w) / TILE;
        inset[seed] = true; key[seed] = 0.0; act.push(seed);
        let (mut processings, mut dispatches) = (0usize, 0usize);

        while !act.is_empty() {
            dispatches += 1;
            // Which tiles run in this dispatch?
            let batch: Vec<usize> = match band {
                None => std::mem::take(&mut act),
                Some(bw) => {
                    // Key is cached at activation time -- recomputing it for every
                    // active tile every dispatch is quadratic, and a GPU version
                    // would compute it once when the tile is queued anyway.
                    let gmin = act.iter().map(|&t| key[t]).fold(f32::INFINITY, f32::min);
                    let cut = gmin + bw;
                    let (take, keep): (Vec<usize>, Vec<usize>) =
                        std::mem::take(&mut act).into_iter().partition(|&t| key[t] < cut);
                    act = keep;
                    take
                }
            };
            sizes.push(batch.len());
            assert!(!batch.is_empty(), "empty batch with {} tiles still active -- \
                    the band gate excluded every tile, scheduler would not terminate", act.len());
            for &t in &batch { inset[t] = false; }
            let mut newly: Vec<usize> = Vec::new();
            let snap = p.u.clone();
            for &t in &batch {
                processings += 1;
                let f = p.process_n(t, &snap, iters, gs);
                if f == 0 { continue; }
                let (tr0, tc0) = (t / p.tc, t % p.tc);
                let mut cand: Vec<usize> = vec![t];
                if f & 2 != 0 && tr0 > 0 { cand.push(t - p.tc); }
                if f & 4 != 0 && tr0 + 1 < p.tr { cand.push(t + p.tc); }
                if f & 8 != 0 && tc0 > 0 { cand.push(t - 1); }
                if f & 16 != 0 && tc0 + 1 < p.tc { cand.push(t + 1); }
                for tt in cand {
                    let k = p.tile_key(tt);
                    if k < key[tt] { key[tt] = k; }
                    if !inset[tt] { inset[tt] = true; newly.push(tt); }
                }
            }
            act.extend(newly);
        }
        (p.u, processings, dispatches, sizes)
    }

    /// Does bucketed tile scheduling reduce tile-processings, and at what cost in
    /// dispatches? Both schedulers are validated against the CPU FSM solver.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_bucket_prototype() {
        // Same field as diag_tile_reactivation so the GPU's activation count is a
        // like-for-like baseline for the prototype's processings.
        let (w, h) = (1025usize, 1025usize);
        let n = w * h;
        let cs = 1.0f32;
        let mut speed = vec![1.0f32; n];
        for (r0, c0) in [(160usize, 220usize), (600, 360), (340, 660), (760, 780)] {
            for r in r0..r0 + 100 { for c in c0..c0 + 100 { speed[r * w + c] = 0.0; } }
        }
        let src = (h / 2) * w + 12;
        speed[src] = 1.0;

        // What the real GPU scheduler costs on this field, for comparison.
        {
            let mut ug = vec![0.0f32; n];
            solve_cold_ms_f32(&[ug.as_mut_ptr() as usize], n, &speed, &[src as u32], &[0, 1], w, h, cs);
            let tcx = w.div_ceil(TILE);
            let mut touched = vec![false; tcx * h.div_ceil(TILE)];
            for i in 0..n {
                if ug[i].is_finite() { touched[(i / w / TILE) * tcx + (i % w) / TILE] = true; }
            }
            println!("\nGPU run above: compare its 'activations=' against {} unavoidable tiles",
                     touched.iter().filter(|b| **b).count());
        }

        let sp64: Vec<f64> = speed.iter().map(|&v| v as f64).collect();
        let reference = crate::fsm::solve_typed::<f64>(&sp64, &[src as u32], w, h, cs as f64);
        let check = |u: &[f32]| -> (usize, f64) {
            let (mut mism, mut mrel) = (0usize, 0.0f64);
            for i in 0..n {
                match (u[i].is_finite(), reference[i].is_finite()) {
                    (true, true) => if reference[i] > 1.0 {
                        mrel = mrel.max(((u[i] as f64) - reference[i]).abs() / reference[i]);
                    },
                    (a, b) if a != b => mism += 1,
                    _ => {}
                }
            }
            (mism, mrel)
        };

        // Workgroups needed to saturate this GPU: one tile = one workgroup of
        // TILE*TILE threads. Below that a dispatch leaves cores idle and the
        // work saving does not turn into a time saving.
        const SATURATE: usize = 1024;
        let stats = |sz: &[usize]| -> (usize, usize) {
            let mut v = sz.to_vec();
            v.sort_unstable();
            (v.iter().sum::<usize>() / v.len().max(1), v[v.len() / 2])
        };

        let (ub, pb, db, sb) = proto_run(w, h, &speed, src, cs, None);
        let (m0, e0) = check(&ub);
        let (mean0, med0) = stats(&sb);
        println!("\n{:<16}{:>12}{:>11}{:>9}{:>9}{:>9}{:>9}{:>10}",
                 "scheduler", "processings", "dispatch", "mismatch", "max_rel",
                 "mean_wg", "med_wg", "starved%");
        let starved = |sz: &[usize]| -> f64 {
            100.0 * sz.iter().filter(|&&x| x < SATURATE).count() as f64 / sz.len() as f64
        };
        println!("{:<16}{:>12}{:>11}{:>9}{:>9.5}{:>9}{:>9}{:>9.0}%",
                 "all-active", pb, db, m0, e0, mean0, med0, starved(&sb));
        for mul in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
            let bw = TILE as f32 * cs * mul;
            let (uu, pp, dd, sz) = proto_run(w, h, &speed, src, cs, Some(bw));
            let (mm, ee) = check(&uu);
            let (mean, med) = stats(&sz);
            println!("{:<16}{:>12}{:>11}{:>9}{:>9.5}{:>9}{:>9}{:>9.0}%   ({:.1}x less work, {:.2}x dispatches)",
                     format!("bucket W={bw:.0}"), pp, dd, mm, ee, mean, med, starved(&sz),
                     pb as f64 / pp as f64, dd as f64 / db as f64);
        }
        println!("\nmean_wg/med_wg = tile-workgroups per dispatch; starved% = dispatches below {SATURATE} \
workgroups.");

        // Production runs k destinations through ONE shared active list, so the
        // occupancy above is not the occupancy that matters. Destinations do not
        // finish together: the shared round loop runs as long as the slowest, and
        // once the fast ones drop out the tail dispatches starve. Simulate that by
        // summing the per-dispatch batch sizes elementwise across k sources rather
        // than assuming they multiply by k.
        let srcs: Vec<usize> = [
            (60usize, 40usize), (200, 900), (900, 60), (980, 980), (512, 20),
            (20, 512), (1000, 512), (512, 1000), (300, 300), (700, 700),
            (300, 700), (700, 300), (100, 512), (900, 512),
        ].iter().map(|&(r, c)| r * w + c).filter(|&i| speed[i] > 0.0).collect();
        let k = srcs.len();
        let combine = |band: Option<f32>| -> (usize, usize, Vec<usize>) {
            let mut total = 0usize;
            let mut prof: Vec<usize> = Vec::new();
            for &sd in &srcs {
                let (_, pp, _, sz) = proto_run(w, h, &speed, sd, cs, band);
                total += pp;
                if prof.len() < sz.len() { prof.resize(sz.len(), 0); }
                for (i, &v) in sz.iter().enumerate() { prof[i] += v; }
            }
            (total, prof.len(), prof)
        };
        println!("\nk={k} destinations sharing one active list (elementwise-summed profile):");
        println!("{:<16}{:>12}{:>11}{:>9}{:>9}{:>10}", "scheduler", "processings", "dispatch", "mean_wg", "med_wg", "starved%");
        let (tb, dbk, pfb) = combine(None);
        let (mb, mdb) = stats(&pfb);
        println!("{:<16}{:>12}{:>11}{:>9}{:>9}{:>9.0}%", "all-active", tb, dbk, mb, mdb, starved(&pfb));
        for mul in [1.0f32, 2.0, 4.0] {
            let bw = TILE as f32 * cs * mul;
            let (tt, dd, pf) = combine(Some(bw));
            let (mm, md) = stats(&pf);
            println!("{:<16}{:>12}{:>11}{:>9}{:>9}{:>9.0}%   ({:.1}x less work, {:.2}x dispatches)",
                     format!("bucket W={bw:.0}"), tt, dd, mm, md, starved(&pf),
                     tb as f64 / tt as f64, dd as f64 / dbk as f64);
        }

        // Does buying convergence INSIDE the tile beat re-activating it?
        //
        // An extra in-tile iteration costs one more pass over 64 cells already in
        // shared memory. A re-activation costs the same 8 passes PLUS the global
        // write-back, the enqueue atomics, a dispatch slot, and reloading the whole
        // (TILE+2)^2 halo block from global memory. So iterations are perhaps an
        // order of magnitude cheaper per unit of convergence -- if they actually
        // reduce activations. `cell_updates` = processings * iters is the compute;
        // `processings` drives the memory traffic and dispatch count.
        //
        // gs=true is raster Gauss-Seidel: the upper bound on what perfect in-tile
        // ordering could buy. It is not directly implementable at 64 threads (the
        // dependency admits only anti-diagonal parallelism: 15 barriers per sweep at
        // ~4/64 threads busy), so it bounds the prize rather than offering it.
        println!("\nin-tile iteration count vs re-activation (k=1, all-active):");
        println!("{:<22}{:>13}{:>14}{:>11}{:>10}", "mode", "processings", "cell_updates", "dispatch", "max_rel");
        for &(iters, gs) in &[(4usize, false), (8, false), (12, false), (16, false), (24, false),
                              (32, false), (8, true), (16, true)] {
            let (uu, pp, dd, _) = proto_run_n(w, h, &speed, src, cs, None, iters, gs);
            let (_, ee) = check(&uu);
            println!("{:<22}{:>13}{:>14}{:>11}{:>10.5}",
                     format!("{} iters{}", iters, if gs { " (G-S bound)" } else { "" }),
                     pp, pp * iters, dd, ee);
        }

        // Open field is the friendly case: the band is a long arc holding many tiles.
        // EdSheeran is tortuous, and a serpentine corridor is the adversarial shape --
        // the wavefront is only as wide as the corridor, so the band may hold too few
        // tiles to fill the machine even with k destinations.
        let mut maze = vec![0.0f32; n];
        let (wall, gap) = (24usize, 64usize);
        for r in 0..h {
            for c in 0..w {
                let band_i = r / gap;
                let open = if r % gap < gap - wall { true }
                           else if band_i % 2 == 0 { c > w - gap } else { c < gap };
                if open { maze[r * w + c] = 1.0; }
            }
        }
        let msrcs: Vec<usize> = (0..14)
            .map(|i| (4usize + i * 3) * w + 4 + i * 5)
            .filter(|&i| maze[i] > 0.0).collect();
        if !msrcs.is_empty() {
            println!("\nserpentine corridor, k={} destinations:", msrcs.len());
            println!("{:<16}{:>12}{:>11}{:>9}{:>9}{:>10}", "scheduler", "processings", "dispatch", "mean_wg", "med_wg", "starved%");
            let mcomb = |band: Option<f32>| -> (usize, usize, Vec<usize>) {
                let (mut total, mut prof) = (0usize, Vec::<usize>::new());
                for &sd in &msrcs {
                    let (_, pp, _, sz) = proto_run(w, h, &maze, sd, cs, band);
                    total += pp;
                    if prof.len() < sz.len() { prof.resize(sz.len(), 0); }
                    for (i, &v) in sz.iter().enumerate() { prof[i] += v; }
                }
                (total, prof.len(), prof)
            };
            let (mt, mdp, mpf) = mcomb(None);
            let (mmn, mmd) = stats(&mpf);
            println!("{:<16}{:>12}{:>11}{:>9}{:>9}{:>9.0}%", "all-active", mt, mdp, mmn, mmd, starved(&mpf));
            for mul in [1.0f32, 2.0, 4.0] {
                let bw = TILE as f32 * cs * mul;
                let (tt, dd, pf) = mcomb(Some(bw));
                let (mm, md) = stats(&pf);
                println!("{:<16}{:>12}{:>11}{:>9}{:>9}{:>9.0}%   ({:.1}x less work, {:.2}x dispatches)",
                         format!("bucket W={bw:.0}"), tt, dd, mm, md, starved(&pf),
                         mt as f64 / tt as f64, dd as f64 / mdp as f64);
            }
        }
    }

    /// Would GroupMarching-style bucketing fix the re-activation problem?
    ///
    /// Kim's GMM updates a whole band of the narrow band at once:
    ///     G = { v : phi(v) <= phi(v_min) + h_min/f_max }
    /// everything inside is provably independent of anything outside. Unlike
    /// NaturalMarchingMethod this needs no acyclic graph -- it is a value-range gate
    /// -- so it survives the tile coarsening that made levels impossible.
    ///
    /// Bucket tiles by min(u) over the tile. For the ordering to be usable, a tile's
    /// defining tiles must land in an EARLIER-or-equal bucket. This counts how often
    /// that fails (would force extra passes), and what it costs in parallelism.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_bucket_ordering() {
        use std::collections::HashSet;
        let (w, h) = (513usize, 513usize);
        let n = w * h;
        let cs = 1.0f32;
        let tc = w.div_ceil(TILE);
        let tr = h.div_ceil(TILE);

        let mut sp = vec![1.0f32; n];
        for (r0, c0) in [(80usize, 110usize), (300, 180), (170, 330), (380, 390)] {
            for r in r0..r0 + 50 {
                for c in c0..c0 + 50 { sp[r * w + c] = 0.0; }
            }
        }
        let src = ((h / 2) * w + 6) as u32;
        sp[src as usize] = 1.0;
        let mut u = vec![0.0f32; n];
        solve_cold_ms_f32(&[u.as_mut_ptr() as usize], n, &sp, &[src], &[0, 1], w, h, cs);

        let parents = |x: usize| -> Vec<usize> {
            if sp[x] <= 0.0 || !u[x].is_finite() { return Vec::new(); }
            let (r, c) = (x / w, x % w);
            let g = |rr: i64, cc: i64| -> (f32, usize) {
                if rr < 0 || cc < 0 || rr >= h as i64 || cc >= w as i64 { (f32::INFINITY, usize::MAX) }
                else { let i = rr as usize * w + cc as usize; (u[i], i) }
            };
            let (lv, li) = g(r as i64, c as i64 - 1);
            let (rv, ri) = g(r as i64, c as i64 + 1);
            let (uv, ui) = g(r as i64 - 1, c as i64);
            let (dv, di) = g(r as i64 + 1, c as i64);
            let (a, ai) = if lv <= rv { (lv, li) } else { (rv, ri) };
            let (b, bi) = if uv <= dv { (uv, ui) } else { (dv, di) };
            let cost = cs / sp[x];
            let (lo, hi) = (a.min(b), a.max(b));
            let mut out = Vec::new();
            if lo + cost <= hi { let p = if a <= b { ai } else { bi }; if p != usize::MAX { out.push(p); } }
            else { if ai != usize::MAX { out.push(ai); } if bi != usize::MAX { out.push(bi); } }
            out
        };

        let nt = tc * tr;
        let mut tmin = vec![f32::INFINITY; nt];
        let mut live = vec![false; nt];
        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        for x in 0..n {
            if !u[x].is_finite() { continue; }
            let tx = (x / w / TILE) * tc + (x % w) / TILE;
            live[tx] = true;
            if u[x] < tmin[tx] { tmin[tx] = u[x]; }
            for p in parents(x) {
                let tp = (p / w / TILE) * tc + (p % w) / TILE;
                if tp != tx { edges.insert((tx as u32, tp as u32)); }
            }
        }
        let live_n = live.iter().filter(|b| **b).count();
        let umax = tmin.iter().cloned().filter(|v| v.is_finite()).fold(0.0f32, f32::max);
        println!("grid {w}x{h} TILE={TILE}  live tiles={live_n}  edges={}  u_max(tile min)={umax:.1}",
                 edges.len());
        println!("current solver: 585,380 activations over 202 rounds = 36x re-activation\n");
        println!("{:>8} {:>8} {:>10} {:>10} {:>12} {:>12}",
                 "band W", "buckets", "avg/bucket", "max/bucket", "violations", "same-bucket");
        // W = TILE*h/f_max is the natural tile analogue of Kim's h_min/f_max.
        for wmul in [0.5f32, 1.0, 2.0, 4.0] {
            let bw = TILE as f32 * cs * wmul;
            let bucket = |t: usize| -> i64 { (tmin[t] / bw).floor() as i64 };
            let nb = (0..nt).filter(|&t| live[t]).map(|t| bucket(t)).max().unwrap_or(0) + 1;
            let mut counts = vec![0usize; nb as usize];
            for t in 0..nt { if live[t] { counts[bucket(t) as usize] += 1; } }
            let nonempty: Vec<usize> = counts.iter().cloned().filter(|c| *c > 0).collect();
            let (mut viol, mut same) = (0usize, 0usize);
            for &(tx, tp) in &edges {
                let (bx, bp) = (bucket(tx as usize), bucket(tp as usize));
                if bp > bx { viol += 1; } else if bp == bx { same += 1; }
            }
            println!("{:>8.1} {:>8} {:>10.1} {:>10} {:>11} ({:>4.1}%) {:>6} ({:>4.1}%)",
                     bw, nonempty.len(),
                     live_n as f64 / nonempty.len() as f64,
                     nonempty.iter().cloned().max().unwrap_or(0),
                     viol, 100.0 * viol as f64 / edges.len() as f64,
                     same, 100.0 * same as f64 / edges.len() as f64);
        }
    }

    /// Is tile-level scheduling well-defined, and how many levels would it need?
    ///
    /// The *vertex* defining graph is acyclic by Lemma 9.3, so NaturalMarchingMethod
    /// levels always exist. Coarsening to tiles can break that: the wave may enter
    /// tile A from B at one place and B from A somewhere else, creating a 2-cycle.
    /// Tiles inside a cycle cannot be given a level and would need repeated passes,
    /// so this counts them -- and reports the level count, which is the dispatch
    /// count a tile-level scheduler would cost.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_tile_levels() {
        use std::collections::{HashMap, HashSet};
        let (w, h) = (513usize, 513usize);
        let n = w * h;
        let cs = 1.0f32;
        let tc = w.div_ceil(TILE);
        let tr = h.div_ceil(TILE);

        let mut sp = vec![1.0f32; n];
        for (r0, c0) in [(80usize, 110usize), (300, 180), (170, 330), (380, 390)] {
            for r in r0..r0 + 50 {
                for c in c0..c0 + 50 { sp[r * w + c] = 0.0; }
            }
        }
        let src = ((h / 2) * w + 6) as u32;
        sp[src as usize] = 1.0;
        let mut u = vec![0.0f32; n];
        solve_cold_ms_f32(&[u.as_mut_ptr() as usize], n, &sp, &[src], &[0, 1], w, h, cs);

        let parents = |x: usize| -> Vec<usize> {
            if sp[x] <= 0.0 || !u[x].is_finite() { return Vec::new(); }
            let (r, c) = (x / w, x % w);
            let g = |rr: i64, cc: i64| -> (f32, usize) {
                if rr < 0 || cc < 0 || rr >= h as i64 || cc >= w as i64 { (f32::INFINITY, usize::MAX) }
                else { let i = rr as usize * w + cc as usize; (u[i], i) }
            };
            let (lv, li) = g(r as i64, c as i64 - 1);
            let (rv, ri) = g(r as i64, c as i64 + 1);
            let (uv, ui) = g(r as i64 - 1, c as i64);
            let (dv, di) = g(r as i64 + 1, c as i64);
            let (a, ai) = if lv <= rv { (lv, li) } else { (rv, ri) };
            let (b, bi) = if uv <= dv { (uv, ui) } else { (dv, di) };
            let cost = cs / sp[x];
            let (lo, hi) = (a.min(b), a.max(b));
            let mut out = Vec::new();
            if lo + cost <= hi { let p = if a <= b { ai } else { bi }; if p != usize::MAX { out.push(p); } }
            else { if ai != usize::MAX { out.push(ai); } if bi != usize::MAX { out.push(bi); } }
            out
        };

        let nt = tc * tr;
        let mut preds: Vec<HashSet<u32>> = vec![HashSet::new(); nt];
        let mut succs: Vec<HashSet<u32>> = vec![HashSet::new(); nt];
        let mut live = vec![false; nt];
        for x in 0..n {
            if !u[x].is_finite() { continue; }
            let tx = (x / w / TILE) * tc + (x % w) / TILE;
            live[tx] = true;
            for p in parents(x) {
                let tp = (p / w / TILE) * tc + (p % w) / TILE;
                if tp != tx { preds[tx].insert(tp as u32); succs[tp].insert(tx as u32); }
            }
        }

        // Kahn: peel tiles whose defining tiles are all already levelled.
        let mut indeg: Vec<usize> = (0..nt).map(|t| preds[t].len()).collect();
        let mut level = vec![usize::MAX; nt];
        let mut q: Vec<u32> = (0..nt).filter(|&t| live[t] && indeg[t] == 0).map(|t| t as u32).collect();
        for &t in &q { level[t as usize] = 0; }
        let mut qi = 0;
        while qi < q.len() {
            let t = q[qi] as usize; qi += 1;
            for &sv in &succs[t] {
                let sx = sv as usize;
                if !live[sx] { continue; }
                level[sx] = level[sx].min(usize::MAX).max(level[t] + 1);
                indeg[sx] -= 1;
                if indeg[sx] == 0 { q.push(sv); }
            }
        }
        let live_n = live.iter().filter(|b| **b).count();
        let levelled = (0..nt).filter(|&t| live[t] && level[t] != usize::MAX).count();
        let cyclic = live_n - levelled;
        let maxlvl = (0..nt).filter(|&t| live[t] && level[t] != usize::MAX)
                            .map(|t| level[t]).max().unwrap_or(0);
        // How big are the cyclic knots?
        let mut deg_hist: HashMap<usize, usize> = HashMap::new();
        for t in 0..nt { if live[t] && level[t] == usize::MAX { *deg_hist.entry(preds[t].len()).or_insert(0) += 1; } }
        println!("grid {w}x{h} TILE={TILE}");
        println!("  live tiles              : {live_n}");
        println!("  levelled (acyclic part) : {levelled}  ({:.1}%)", 100.0*levelled as f64/live_n as f64);
        println!("  stuck in cycles         : {cyclic}  ({:.1}%)", 100.0*cyclic as f64/live_n as f64);
        println!("  levels needed (dispatches): {}", maxlvl + 1);
        let mut ks: Vec<_> = deg_hist.into_iter().collect(); ks.sort();
        println!("  cyclic tiles by in-degree: {:?}", &ks[..ks.len().min(6)]);
    }

    /// Stability of the defining graph between consecutive solves.
    ///
    /// Zoennchen's similarity metric (9.51):
    ///     D(G_i, G_i+1) = (|E_i \ E_i+1| + |E_i+1 \ E_i|) / (|E_i| + |E_i+1|)
    /// IFIM and NaturalMarchingMethod both rest on the assumption D ~ 0, i.e. the
    /// wave arrives in essentially the same order from one solve to the next. If the
    /// graph churns, reusing the previous order schedules work too late, which
    /// stalls propagation just as badly as scheduling it too early (Sec. 9.5.1).
    ///
    /// Reported at both vertex granularity (as in the thesis) and tile granularity
    /// (what a tile-level scheduler on this solver would actually reuse).
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_defining_graph_similarity() {
        use std::collections::HashSet;
        let (w, h) = (513usize, 513usize);
        let n = w * h;
        let cs = 1.0f32;
        let tc = w.div_ceil(TILE);

        let mut base = vec![1.0f32; n];
        for (r0, c0) in [(80usize, 110usize), (300, 180), (170, 330), (380, 390)] {
            for r in r0..r0 + 50 {
                for c in c0..c0 + 50 {
                    base[r * w + c] = 0.0;
                }
            }
        }
        let src = ((h / 2) * w + 6) as u32;
        base[src as usize] = 1.0;

        // Upwind neighbours of cell x, per the Godunov update actually used.
        let parents = |u: &[f32], sp: &[f32], x: usize| -> Vec<usize> {
            if sp[x] <= 0.0 || !u[x].is_finite() {
                return Vec::new();
            }
            let (r, c) = (x / w, x % w);
            let g = |rr: i64, cc: i64| -> (f32, usize) {
                if rr < 0 || cc < 0 || rr >= h as i64 || cc >= w as i64 {
                    (f32::INFINITY, usize::MAX)
                } else {
                    let i = rr as usize * w + cc as usize;
                    (u[i], i)
                }
            };
            let (lv, li) = g(r as i64, c as i64 - 1);
            let (rv, ri) = g(r as i64, c as i64 + 1);
            let (uv, ui) = g(r as i64 - 1, c as i64);
            let (dv, di) = g(r as i64 + 1, c as i64);
            let (a, ai) = if lv <= rv { (lv, li) } else { (rv, ri) };
            let (b, bi) = if uv <= dv { (uv, ui) } else { (dv, di) };
            let cost = cs / sp[x];
            let (lo, hi) = (a.min(b), a.max(b));
            let mut out = Vec::new();
            if lo + cost <= hi {
                let p = if a <= b { ai } else { bi };
                if p != usize::MAX { out.push(p); }
            } else {
                if ai != usize::MAX { out.push(ai); }
                if bi != usize::MAX { out.push(bi); }
            }
            out
        };
        let graphs = |u: &[f32], sp: &[f32]| -> (HashSet<(u32, u32)>, HashSet<(u32, u32)>) {
            let (mut vg, mut tg) = (HashSet::new(), HashSet::new());
            for x in 0..n {
                let tx = ((x / w / TILE) * tc + (x % w) / TILE) as u32;
                for p in parents(u, sp, x) {
                    vg.insert((x as u32, p as u32));
                    let tp = ((p / w / TILE) * tc + (p % w) / TILE) as u32;
                    if tp != tx { tg.insert((tx, tp)); }
                }
            }
            (vg, tg)
        };
        let d = |a: &HashSet<(u32, u32)>, b: &HashSet<(u32, u32)>| -> f64 {
            let diff = a.difference(b).count() + b.difference(a).count();
            diff as f64 / (a.len() + b.len()).max(1) as f64
        };

        // A crowd that slows a patch and drifts, as in a dynamic navigation field.
        let solve = |sp: &[f32]| -> Vec<f32> {
            let mut u = vec![0.0f32; n];
            solve_cold_ms_f32(&[u.as_mut_ptr() as usize], n, sp, &[src], &[0, 1], w, h, cs);
            u
        };
        let field_at = |step: usize| -> Vec<f32> {
            let mut sp = base.clone();
            let (cr, cc) = (200usize + step * 6, 150usize + step * 12);
            for r in cr..(cr + 60).min(h) {
                for c in cc..(cc + 60).min(w) {
                    if sp[r * w + c] > 0.0 { sp[r * w + c] = 0.35; }
                }
            }
            sp
        };

        let mut prev: Option<(HashSet<(u32,u32)>, HashSet<(u32,u32)>)> = None;
        println!("{:>5}  {:>12}  {:>12}  {:>10}  {:>10}", "step", "|E_vertex|", "|E_tile|", "D_vertex", "D_tile");
        for step in 0..6 {
            let sp = field_at(step);
            let u = solve(&sp);
            let (vg, tg) = graphs(&u, &sp);
            if let Some((pv, pt)) = &prev {
                println!("{step:>5}  {:>12}  {:>12}  {:>10.4}  {:>10.4}",
                         vg.len(), tg.len(), d(pv, &vg), d(pt, &tg));
            } else {
                println!("{step:>5}  {:>12}  {:>12}  {:>10}  {:>10}", vg.len(), tg.len(), "-", "-");
            }
            prev = Some((vg, tg));
        }
    }

    /// How many times is a tile re-activated per solve?
    ///
    /// This is the quantity IFIM minimises. FMM is optimal at exactly one update per
    /// vertex; FIM re-updates as the band sloshes. Zoennchen reports FIM needing
    /// 75k-193k updates against u_FMM = 53,888 (~2-4x) on Richard-Wagner-Strasse.
    /// If our tiles already activate close to once, there is nothing to recover.
    ///
    /// Run with EIKONAL_RPB=1 EIKONAL_BREAKDOWN=1 so every round is sampled.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_tile_reactivation() {
        let (w, h) = (1025usize, 1025usize);
        let n = w * h;
        let mut speed = vec![1.0f32; n];
        for (r0, c0) in [(160usize, 220usize), (600, 360), (340, 660), (760, 780)] {
            for r in r0..r0 + 100 {
                for c in c0..c0 + 100 {
                    speed[r * w + c] = 0.0;
                }
            }
        }
        let src = ((h / 2) * w + 12) as u32;
        speed[src as usize] = 1.0;

        let mut u = vec![0.0f32; n];
        solve_cold_ms_f32(&[u.as_mut_ptr() as usize], n, &speed, &[src], &[0, 1], w, h, 1.0);

        // A tile is unavoidable if it holds at least one reachable cell: any correct
        // solver must process it at least once. That is the FMM-optimal baseline.
        let tc = w.div_ceil(TILE);
        let tr = h.div_ceil(TILE);
        let mut touched = vec![false; tc * tr];
        for i in 0..n {
            if u[i].is_finite() {
                touched[(i / w / TILE) * tc + (i % w) / TILE] = true;
            }
        }
        let unavoidable = touched.iter().filter(|b| **b).count();
        println!(
            "grid {w}x{h} TILE={TILE}  tiles={}  tiles_with_reachable_cells={unavoidable}",
            tc * tr
        );
        println!("  -> compare 'activations' printed above against {unavoidable}");
    }

    /// Feasibility check for dependency-cone ("informed FIM") invalidation.
    ///
    /// The threshold reset discards every cell with u >= min(prior over changed),
    /// which is the whole annulus outside that contour. The dependency cone is the
    /// subset of that annulus actually reachable along characteristics from a
    /// changed cell -- a shadow behind the change rather than a full ring. This
    /// measures the ratio, which decides whether cone tracking is worth building.
    ///
    /// Parents are derived from u, not stored: for the 2-D Godunov update
    /// a = min(u[x-1], u[x+1]), b = min(u[y-1], u[y+1]), the upwind neighbours are
    /// the argmins -- one when lo + cost <= hi, otherwise both.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cone_vs_threshold() {
        let (w, h) = (513usize, 513usize);
        let n = w * h;
        let cs = 1.0f32;
        let mut speed = vec![1.0f32; n];
        for (r0, c0) in [(80usize, 110usize), (300, 180), (170, 330), (380, 390)] {
            for r in r0..r0 + 50 {
                for c in c0..c0 + 50 {
                    speed[r * w + c] = 0.0;
                }
            }
        }
        let src = ((h / 2) * w + 6) as u32;
        speed[src as usize] = 1.0;

        let mut u = vec![0.0f32; n];
        solve_cold_ms_f32(&[u.as_mut_ptr() as usize], n, &speed, &[src], &[0, 1], w, h, cs);

        // Is `p` an upwind parent of cell `x`?
        let is_parent = |x: usize, p: usize| -> bool {
            let (r, c) = (x / w, x % w);
            let g = |rr: i64, cc: i64| -> (f32, usize) {
                if rr < 0 || cc < 0 || rr >= h as i64 || cc >= w as i64 {
                    (f32::INFINITY, usize::MAX)
                } else {
                    let i = rr as usize * w + cc as usize;
                    (u[i], i)
                }
            };
            let (lv, li) = g(r as i64, c as i64 - 1);
            let (rv, ri) = g(r as i64, c as i64 + 1);
            let (uv, ui) = g(r as i64 - 1, c as i64);
            let (dv, di) = g(r as i64 + 1, c as i64);
            let (a, ai) = if lv <= rv { (lv, li) } else { (rv, ri) };
            let (b, bi) = if uv <= dv { (uv, ui) } else { (dv, di) };
            if speed[x] <= 0.0 {
                return false;
            }
            let cost = cs / speed[x];
            let (lo, hi) = (a.min(b), a.max(b));
            if lo + cost <= hi {
                // one-sided: only the smaller axis contributes
                (if a <= b { ai } else { bi }) == p
            } else {
                ai == p || bi == p
            }
        };

        for (label, r0, c0) in [("change near source", 250usize, 40usize),
                                ("change mid field",   250, 250),
                                ("change far field",   250, 450)] {
            let mut changed = Vec::new();
            for r in r0..(r0 + 12).min(h) {
                for c in c0..(c0 + 12).min(w) {
                    if speed[r * w + c] > 0.0 {
                        changed.push(r * w + c);
                    }
                }
            }
            let thr = changed.iter().map(|&c| u[c]).fold(f32::INFINITY, f32::min);
            let thr_region = u.iter().filter(|v| v.is_finite() && **v >= thr).count();

            // BFS the cone: a neighbour x downstream of m joins if m is its parent.
            let mut marked = vec![false; n];
            let mut q: Vec<usize> = Vec::new();
            for &c in &changed { if !marked[c] { marked[c] = true; q.push(c); } }
            let mut qi = 0;
            while qi < q.len() {
                let m = q[qi]; qi += 1;
                let (r, c) = (m / w, m % w);
                let mut nb = Vec::new();
                if c > 0 { nb.push(m - 1); }
                if c + 1 < w { nb.push(m + 1); }
                if r > 0 { nb.push(m - w); }
                if r + 1 < h { nb.push(m + w); }
                for x in nb {
                    if marked[x] || !u[x].is_finite() || u[x] <= u[m] { continue; }
                    if is_parent(x, m) { marked[x] = true; q.push(x); }
                }
            }
            let cone = marked.iter().filter(|b| **b).count();
            let finite = u.iter().filter(|v| v.is_finite()).count();
            println!(
                "{label:20} thr={thr:7.1}  threshold_region={thr_region:>7} ({:5.1}% of field)  \
cone={cone:>7} ({:5.1}%)  cone/threshold={:.3}",
                100.0 * thr_region as f64 / finite as f64,
                100.0 * cone as f64 / finite as f64,
                cone as f64 / thr_region.max(1) as f64
            );
        }
    }

    /// Strict warm-vs-cold equivalence. A warm restart is only useful if it is
    /// indistinguishable from a cold solve on the new speed field, so every case
    /// here compares against `solve_cold_ms_f32` on exactly that field.
    ///
    /// The cases are chosen to cover the failure mode the monotone update rule has:
    /// when speed DROPS the true travel time RISES, and FIM can only lower values,
    /// so a stale prior is a fixed point unless the threshold reset discards it.
    /// Distance from the source matters too -- it sets how much of the field the
    /// threshold keeps.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_warm_strict_vs_cold() {
        let (w, h) = (193usize, 193usize);
        let n = w * h;
        let src = ((h / 2) * w + 8) as u32;

        // A few scattered blocks so paths bend around obstacles.
        let mut base = vec![1.0f32; n];
        for (r0, c0) in [(30usize, 40usize), (120, 60), (60, 120), (140, 140)] {
            for r in r0..r0 + 20 {
                for c in c0..c0 + 20 {
                    base[r * w + c] = 0.0;
                }
            }
        }
        base[src as usize] = 1.0;

        let cases: &[(&str, usize, usize, f32)] = &[
            ("slowdown near source", 40, 20, 0.25),
            ("slowdown mid field",   90, 90, 0.25),
            ("slowdown far corner", 160, 160, 0.25),
            ("speedup mid field",    90, 90, 4.0),
            ("severe slowdown",      90, 60, 0.05),
        ];

        let mut worst = 0.0f64;
        for (label, r0, c0, factor) in cases {
            let mut prior = vec![0.0f32; n];
            solve_cold_ms_f32(&[prior.as_mut_ptr() as usize], n, &base, &[src], &[0, 1], w, h, 1.0);

            let mut newspeed = base.clone();
            for r in *r0..(*r0 + 18).min(h) {
                for c in *c0..(*c0 + 18).min(w) {
                    if newspeed[r * w + c] > 0.0 {
                        newspeed[r * w + c] = *factor;
                    }
                }
            }
            let changed: Vec<u32> = (0..n)
                .filter(|&i| (newspeed[i] - base[i]).abs() > 1e-12)
                .map(|i| i as u32)
                .collect();

            let mut warm = vec![0.0f32; n];
            solve_warm_ms_f32(&[warm.as_mut_ptr() as usize], &[prior.as_ptr() as usize],
                              n, &newspeed, &changed, &[src], &[0, 1], w, h, 1.0);
            let mut cold = vec![0.0f32; n];
            solve_cold_ms_f32(&[cold.as_mut_ptr() as usize], n, &newspeed, &[src], &[0, 1], w, h, 1.0);

            let (mut mism, mut lo, mut hi) = (0usize, 0.0f64, 0.0f64);
            let mut max_rel = 0.0f64;
            for i in 0..n {
                match (warm[i].is_finite(), cold[i].is_finite()) {
                    (true, true) => {
                        let d = (warm[i] - cold[i]) as f64;
                        if d < lo { lo = d; }
                        if d > hi { hi = d; }
                        if cold[i] > 1.0 {
                            max_rel = max_rel.max(d.abs() / cold[i] as f64);
                        }
                    }
                    (a, b) if a != b => mism += 1,
                    _ => {}
                }
            }
            worst = worst.max(max_rel);
            println!("{label:22} changed={:5}  reach_mismatch={mism:5}  \
warm-cold in [{lo:+.3},{hi:+.3}]  max_rel={max_rel:.5}", changed.len());
            assert_eq!(mism, 0, "{label}: warm/cold reachability differs on {mism} cells");
        }
        println!("worst relative error across all cases: {worst:.5}");
        assert!(worst < 0.02, "warm deviates from cold by {worst:.5} relative");
    }

    /// Obstacle-rich cross-check against the CPU FSM solver. This is the real gate
    /// for the tile-activation rule: staggered barriers force the wavefront to wrap
    /// around walls, so information reaches a tile from directions a smooth frontier
    /// never exercises. If directional neighbour activation ever misses an edge, the
    /// reachable set or the travel times diverge from FSM here.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_obstacles_vs_cpu() {
        for &w in &[257usize, 513] {
            let h = w;
            let n = w * h;
            // Scattered rectangular blocks: forces the wavefront to wrap around
            // obstacles from several directions without making the geodesic path
            // pathologically longer than the diagonal (which would exhaust
            // max_rounds -- see the tortuosity note in solve_cold_ms_f32).
            let mut speed = vec![1.0f32; n];
            let bs = 9usize;
            let mut r0 = 12usize;
            let mut toggle = 0usize;
            while r0 + bs < h {
                let mut c0 = 12 + (toggle % 2) * 20;
                while c0 + bs < w {
                    for r in r0..r0 + bs {
                        for c in c0..c0 + bs {
                            speed[r * w + c] = 0.0;
                        }
                    }
                    c0 += 40;
                }
                r0 += 28;
                toggle += 1;
            }
            let src = ((h / 2) * w + 1) as u32;
            speed[src as usize] = 1.0;

            let mut gpu = vec![0.0f32; n];
            let ptrs = vec![gpu.as_mut_ptr() as usize];
            solve_cold_ms_f32(&ptrs, n, &speed, &[src], &[0, 1], w, h, 1.0);

            let speed64: Vec<f64> = speed.iter().map(|&v| v as f64).collect();
            let cpu = crate::fsm::solve_typed::<f64>(&speed64, &[src], w, h, 1.0);

            let (mut only_gpu, mut only_cpu, mut both) = (0usize, 0usize, 0usize);
            let (mut max_abs, mut max_rel, mut umax) = (0.0f64, 0.0f64, 0.0f64);
            let mut checksum = 0.0f64;
            for i in 0..n {
                let g = gpu[i] as f64;
                match (g.is_finite(), cpu[i].is_finite()) {
                    (true, true) => {
                        both += 1;
                        checksum += g;
                        umax = umax.max(cpu[i]);
                        max_abs = max_abs.max((g - cpu[i]).abs());
                        if cpu[i] > 1.0 {
                            max_rel = max_rel.max((g - cpu[i]).abs() / cpu[i]);
                        }
                    }
                    (true, false) => only_gpu += 1,
                    (false, true) => only_cpu += 1,
                    _ => {}
                }
            }
            println!(
                "grid {w}x{h}: both={both} gpu_only={only_gpu} cpu_only={only_cpu} \
max_abs={max_abs:.3} max_rel={max_rel:.5} u_max={umax:.1} checksum={checksum:.1}"
            );
            // Reachability must match exactly -- that is what the tile-activation
            // rule governs. Values differ slightly because the GPU skips updates
            // below conv_tol, which accumulates along long paths, so compare
            // relatively rather than absolutely.
            assert_eq!(only_cpu, 0, "GPU failed to reach {only_cpu} cells FSM reached");
            assert_eq!(only_gpu, 0, "GPU reached {only_gpu} cells FSM did not");
            assert!(max_rel < 0.02, "GPU vs FSM max relative error {max_rel:.5} too large");
        }
    }

    /// Cold GPU solve on a uniform field with one central source: every cell is
    /// reachable, so u must be finite everywhere and equal the Euclidean distance.
    #[test]
    #[ignore = "manual diagnostic"]
    fn diag_cold_convergence() {
        for &w in &[201usize, 501, 1001, 2001] {
            let h = w;
            let n = w * h;
            let src = ((h / 2) * w + w / 2) as u32;
            let speed = vec![1.0f32; n];
            let mut out = vec![0.0f32; n];
            let ptrs = vec![out.as_mut_ptr() as usize];
            solve_cold_ms_f32(&ptrs, n, &speed, &[src], &[0, 1], w, h, 1.0);

            let finite = out.iter().filter(|v| v.is_finite()).count();
            let (cy, cx) = ((h / 2) as f64, (w / 2) as f64);
            let mut max_rel = 0.0f64;
            for i in 0..h {
                for j in 0..w {
                    let v = out[i * w + j];
                    let exp = (((i as f64 - cy).powi(2)) + ((j as f64 - cx).powi(2))).sqrt();
                    if v.is_finite() && exp > 5.0 {
                        max_rel = max_rel.max(((v as f64 - exp) / exp).abs());
                    }
                }
            }
            println!(
                "grid {w:>5}x{h:<5} finite {finite:>10}/{n:<10} ({:>5.1}%)  max_rel_err {max_rel:.4}",
                finite as f64 / n as f64 * 100.0
            );
        }
    }

    /// Manual cold-solve benchmark on the production grid size.
    ///   cargo test --release bench_cold_k16 -- --ignored --nocapture
    ///   EIKONAL_BENCH_K=4 cargo test --release bench_cold_k16 -- --ignored --nocapture
    ///
    /// `finite` in the checksum line must equal k*n. It does not for k >= 8: once
    /// the active-tile count exceeds max_compute_workgroups_per_dimension (65535)
    /// the indirect dispatch is clamped and the dropped tiles are lost for good,
    /// because try_enqueue already bumped their tile_round tag.
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_cold_k16() {
        let (w, h) = (4107usize, 3769usize);
        let n = w * h;
        let k: usize = std::env::var("EIKONAL_BENCH_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);

        let speed = vec![1.0f32; n];
        let sources: Vec<u32> = (0..k)
            .map(|d| {
                let r = (h / (k + 1)) * (d + 1);
                let c = (w / (k + 1)) * (d + 1);
                (r * w + c) as u32
            })
            .collect();
        let src_offsets: Vec<u32> = (0..=k as u32).collect();

        let mut outs: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; n]).collect();
        let out_ptrs: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();

        println!("\ngrid = {w}x{h}  n = {n}  k = {k}");

        let iters = 5;
        let mut ms = Vec::new();
        for i in 0..iters {
            let t = std::time::Instant::now();
            solve_cold_ms_f32(&out_ptrs, n, &speed, &sources, &src_offsets, w, h, 0.2);
            let e = t.elapsed().as_secs_f64() * 1e3;
            ms.push(e);
            println!("  iter {i}: {e:8.1} ms");
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  median: {:.1} ms   best: {:.1} ms", ms[iters / 2], ms[0]);

        // Checksum over all destinations — must be identical across both modes.
        let mut finite = 0u64;
        let mut sum = 0.0f64;
        for o in &outs {
            for &v in o.iter() {
                if v.is_finite() {
                    finite += 1;
                    sum += v as f64;
                }
            }
        }
        println!("  checksum: finite={finite} sum={sum:.3}");
    }

    fn gpu_solve_cold(speed: &[f64], source: u32, w: usize, h: usize) -> Vec<f64> {
        let n = w * h;
        let mut out = vec![0.0f64; n];
        let ptr = out.as_mut_ptr() as usize;
        solve_cold(&[ptr], n, speed, &[source], w, h, 1.0);
        out
    }

    fn gpu_solve_warm(
        speed: &[f64],
        prior: &[f64],
        changed: &[u32],
        source: u32,
        w: usize,
        h: usize,
    ) -> Vec<f64> {
        let n = w * h;
        let mut out = vec![0.0f64; n];
        let out_ptr = out.as_mut_ptr() as usize;
        let prior_ptr = prior.as_ptr() as usize;
        solve_warm(
            &[out_ptr],
            &[prior_ptr],
            n,
            speed,
            changed,
            &[source],
            w,
            h,
            1.0,
        );
        out
    }

    #[test]
    fn warm_gpu_increases_travel_time_on_speed_drop() {
        let w = 51usize;
        let h = 51usize;
        let n = w * h;
        let source = (25 * w + 25) as u32;

        let speed_old = vec![1.0f64; n];

        let cold_old = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu_solve_cold(&speed_old, source, w, h)
        })) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("[skip] warm_gpu_increases_travel_time_on_speed_drop: no GPU available");
                return;
            }
        };

        let mut speed_new = speed_old.clone();
        for r in 35..42 {
            for c in 35..42 {
                speed_new[r * w + c] = 0.3;
            }
        }
        let changed: Vec<u32> = (0..n)
            .filter(|&i| (speed_new[i] - speed_old[i]).abs() > 1e-12)
            .map(|i| i as u32)
            .collect();

        let warm = gpu_solve_warm(&speed_new, &cold_old, &changed, source, w, h);
        let cold_new = gpu_solve_cold(&speed_new, source, w, h);

        let mut max_err = 0.0f64;
        for i in 0..n {
            if cold_new[i].is_finite() {
                max_err = max_err.max((warm[i] - cold_new[i]).abs());
            }
        }
        assert!(
            max_err < 10.0 * CONV_TOL as f64,
            "GPU warm vs cold max error {max_err:.4} exceeds 10×CONV_TOL={:.4}",
            10.0 * CONV_TOL as f64,
        );
    }
}




