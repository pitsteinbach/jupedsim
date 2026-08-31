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

const TILE: usize = 16;
const SENTINEL: f32 = 1e30_f32;
const CONV_TOL: f32 = 1e-2;

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
    _pad: [u32; 2], // row 2 — 16 bytes; total 48 bytes
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

// ── Device-level context (created once, shared across all solver instances) ───

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fim_pipeline: wgpu::ComputePipeline,
    setup_pipeline: wgpu::ComputePipeline,
    fill_pipeline: wgpu::ComputePipeline,
    fim_bgl_0: wgpu::BindGroupLayout,
    fim_bgl_1: wgpu::BindGroupLayout,
    setup_bgl_0: wgpu::BindGroupLayout,
    fill_bgl_0: wgpu::BindGroupLayout,
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
        let native_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: native_limits,
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .expect("failed to open GPU device");

        // ── Shaders ───────────────────────────────────────────────────────────
        let fim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fim_update.wgsl").into()),
        });
        let setup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/indirect_setup.wgsl").into()),
        });
        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fill_sentinel.wgsl").into()),
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
            entries: &[bgl_entry(0, storage_rw()), bgl_entry(1, storage_rw())],
        });
        // Fill group 0: u(rw) — sentinel fill shader
        let fill_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_entry(0, storage_rw())],
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

        GpuContext {
            device,
            queue,
            fim_pipeline,
            setup_pipeline,
            fill_pipeline,
            fim_bgl_0,
            fim_bgl_1,
            setup_bgl_0,
            fill_bgl_0,
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
    params_buf: wgpu::Buffer,
    // 4-byte staging buffer used to read back indirect_buf[0] for early termination.
    count_staging: wgpu::Buffer,
    // One MAP_READ staging buffer per destination (N × 4 bytes each, SHARED).
    // Persisted across solves so readback can submit all k copies in a single
    // command encoder and poll once instead of one poll per 32 MB chunk.
    staging: Vec<wgpu::Buffer>,

    // Bind groups
    bg_persistent: wgpu::BindGroup,
    bg_ping: wgpu::BindGroup,
    bg_pong: wgpu::BindGroup,
    setup_bg_ping: wgpu::BindGroup,
    setup_bg_pong: wgpu::BindGroup,
    fill_bg: wgpu::BindGroup,

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
            _pad: [0u32; 2],
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
                | wgpu::BufferUsages::COPY_DST,
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
        // 4-byte read-back buffer for the active-tile count.  Reused across
        // solves; always unmapped before the next batch copy is issued.
        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // ── Per-destination readback staging (SHARED, MAP_READ) ──────────────
        // k buffers of N × 4 bytes each.  Sized to hold one full destination
        // slice so readback can submit all k GPU copies in a single encoder.
        let staging: Vec<wgpu::Buffer> = (0..k)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (n * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

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
            ],
        });

        GpuFimSolver {
            u_buf,
            speed_buf,
            sources_buf,
            tile_round_buf,
            active_a,
            active_b,
            count_a,
            count_b,
            indirect_buf,
            params_buf,
            staging,
            count_staging,
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
        }
    }

    // ── Per-solve uploads ─────────────────────────────────────────────────────

    fn upload_speed_and_sources(
        &self,
        ctx: &GpuContext,
        speed_field: &[f64],
        sources: &[u32],
        cell_size: f64,
    ) {
        let speed_f32: Vec<f32> = speed_field.iter().map(|&v| v as f32).collect();
        ctx.queue
            .write_buffer(&self.speed_buf, 0, f32_as_bytes(&speed_f32));
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
            _pad: [0u32; 2],
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
            _pad: [0u32; 2],
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, unsafe { as_bytes(&params) });
    }

    /// Reads back GPU travel times into caller-supplied `*mut f32` buffers.
    /// Replaces sentinel values with `f32::INFINITY`; no precision conversion needed.
    fn readback_f32(&self, ctx: &GpuContext, out_ptrs: &[usize]) {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for (d, stg) in self.staging.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                &self.u_buf,
                (d * self.n) as u64 * 4,
                stg,
                0,
                (self.n * 4) as u64,
            );
        }
        let si = ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(si),
            timeout: None,
        });

        for stg in &self.staging {
            stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

        for (stg, &ptr) in self.staging.iter().zip(out_ptrs.iter()) {
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, self.n) };
            let mapped = stg.slice(..).get_mapped_range();
            let u_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(mapped.as_ptr() as *const f32, self.n) };
            for (d, &v) in dst.iter_mut().zip(u_f32.iter()) {
                *d = if v < SENTINEL * 0.5 { v } else { f32::INFINITY };
            }
            drop(mapped);
            stg.unmap();
        }
    }

    fn reset_tile_round(&self, ctx: &GpuContext) {
        let zeros = vec![0u8; self.k * self.num_tiles * 4];
        ctx.queue.write_buffer(&self.tile_round_buf, 0, &zeros);
    }

    fn gpu_fill_sentinel(&self, ctx: &GpuContext) {
        // Fill u_buf with SENTINEL entirely on the GPU at ~200 GB/s instead of
        // writing k×N×4 bytes through the Metal SHARED heap (~1 GB/s).
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

    fn init_u_cold(&self, ctx: &GpuContext, sources: &[u32]) {
        self.gpu_fill_sentinel(ctx);
        for (d, &src) in sources.iter().enumerate() {
            self.write_source_cells(ctx, d, &[src]);
        }
    }

    fn init_u_cold_ms(&self, ctx: &GpuContext, sources_flat: &[u32], src_offsets: &[u32]) {
        self.gpu_fill_sentinel(ctx);
        for d in 0..self.k {
            let srcs = &sources_flat[src_offsets[d] as usize..src_offsets[d + 1] as usize];
            self.write_source_cells(ctx, d, srcs);
        }
    }

    fn set_source_cells(&self, ctx: &GpuContext, sources: &[u32]) {
        for (d, &src) in sources.iter().enumerate() {
            self.write_source_cells(ctx, d, &[src]);
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

    fn seed_slots_cold(&self, sources: &[u32]) -> Vec<u32> {
        let mut slots = HashSet::new();
        for (d, &src) in sources.iter().enumerate() {
            self.seed_tiles_for_dest(d, &[src], &mut slots);
        }
        slots.into_iter().collect()
    }

    fn seed_slots_cold_ms(&self, sources_flat: &[u32], src_offsets: &[u32]) -> Vec<u32> {
        let mut slots = HashSet::new();
        for d in 0..self.k {
            let srcs = &sources_flat[src_offsets[d] as usize..src_offsets[d + 1] as usize];
            self.seed_tiles_for_dest(d, srcs, &mut slots);
        }
        slots.into_iter().collect()
    }

    fn seed_slots_warm(&self, changed_cells: &[u32], sources: &[u32]) -> Vec<u32> {
        let mut slots = HashSet::new();
        for (d, &src) in sources.iter().enumerate() {
            self.seed_tiles_for_dest(d, &[src], &mut slots);
            for &cell in changed_cells {
                let tile_r = (cell as usize / self.width) / TILE;
                let tile_c = (cell as usize % self.width) / TILE;
                self.add_tile_and_neighbors(d, tile_r, tile_c, &mut slots);
            }
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
        let indirect_init: Vec<u8> = [cnt, 1u32, 1u32]
            .iter()
            .flat_map(|u| u.to_ne_bytes())
            .collect();
        ctx.queue
            .write_buffer(&self.indirect_buf, 0, &indirect_init);
        for &slot in slots {
            let byte_offset = (slot as usize * 4) as u64;
            ctx.queue
                .write_buffer(&self.tile_round_buf, byte_offset, &1u32.to_ne_bytes());
        }
    }

    // ── Encode and submit ─────────────────────────────────────────────────────

    fn encode_and_submit(&self, ctx: &GpuContext) {
        // Split rounds into fixed-size batches (one Metal command buffer each).
        // Large grids require 1000+ rounds; a single command buffer that large
        // can exceed Metal's internal command-complexity limit.
        //
        // After each batch the last setup_indirect call has written the active-tile
        // count for the *next* round into indirect_buf[0].  Copying that 4-byte
        // value to count_staging and reading it back lets the CPU stop submitting
        // new batches as soon as FIM converges — a big win for warm restarts where
        // only a small region changed.
        const ROUNDS_PER_BATCH: u32 = 64;

        let mut round = 0u32;
        while round < self.max_rounds {
            let batch_end = (round + ROUNDS_PER_BATCH).min(self.max_rounds);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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

            // Append the count copy inside the same encoder so it executes
            // immediately after the last setup call with no extra submission.
            enc.copy_buffer_to_buffer(&self.indirect_buf, 0, &self.count_staging, 0, 4);

            let si = ctx.queue.submit([enc.finish()]);
            // map_async before poll: the Wait poll both completes the submission
            // and fires the map callback in one call.
            self.count_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::Wait {
                submission_index: Some(si),
                timeout: None,
            });

            let active_next = {
                let mapped = self.count_staging.slice(..).get_mapped_range();
                let count = u32::from_ne_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
                drop(mapped);
                self.count_staging.unmap();
                count
            };
            if crate::GPU_EARLY_TERMINATION && active_next == 0 {
                break;
            }
        }
    }

    // ── Readback ──────────────────────────────────────────────────────────────

    fn readback(&self, ctx: &GpuContext, out_ptrs: &[usize]) {
        use rayon::prelude::*;

        // All k destination slices are copied in one command encoder so there is
        // exactly ONE GPU→CPU poll instead of one per 32 MB chunk.  The staging
        // buffers are persistent (allocated in GpuFimSolver::new) so there is
        // no per-call allocation overhead either.

        let t_total = crate::PRINT_TIMINGS.then(std::time::Instant::now);

        // ── GPU phase: copy all k slices of u_buf into the staging buffers ────
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for (d, stg) in self.staging.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                &self.u_buf,
                (d * self.n) as u64 * 4,
                stg,
                0,
                (self.n * 4) as u64,
            );
        }
        let si = ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(si),
            timeout: None,
        });

        let t_after_copy = crate::PRINT_TIMINGS.then(std::time::Instant::now);

        // ── CPU phase: map all, convert in parallel, unmap ───────────────────
        // Data is already in SHARED memory; map_async callbacks fire in one poll.
        for stg in &self.staging {
            stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

        // BufferView is !Send.  Extract raw f32 pointers before crossing thread
        // boundaries; the underlying Metal SHARED pages are valid until unmap().
        let mapped: Vec<_> = self
            .staging
            .iter()
            .map(|stg| stg.slice(..).get_mapped_range())
            .collect();

        struct SendPtr(*const f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let src_ptrs: Vec<_> = mapped
            .iter()
            .map(|m| SendPtr(m.as_ptr() as *const f32))
            .collect();

        let n = self.n;
        // Outer parallelism: one rayon task per destination (k-way).
        // Inner parallelism: par_chunks_mut across the N-element slice, so
        // k=1 also saturates multiple cores on large grids.
        const CHUNK: usize = 1 << 18; // 1 MB of f32 input per chunk
        src_ptrs
            .par_iter()
            .zip(out_ptrs.par_iter())
            .for_each(|(src, &dst_ptr)| {
                let src_f32 = unsafe { std::slice::from_raw_parts(src.0, n) };
                let dst_f64 = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f64, n) };
                dst_f64
                    .par_chunks_mut(CHUNK)
                    .zip(src_f32.par_chunks(CHUNK))
                    .for_each(|(d, s)| {
                        for (&v, out) in s.iter().zip(d.iter_mut()) {
                            let as_f64 = v as f64;
                            *out = if v < SENTINEL * 0.5 {
                                as_f64
                            } else {
                                f64::INFINITY
                            };
                        }
                    });
            });

        drop(mapped);
        for stg in &self.staging {
            stg.unmap();
        }

        if let Some(t) = t_total {
            let total_ms = t.elapsed().as_secs_f64() * 1e3;
            let convert_ms = t_after_copy
                .map(|t2| t2.elapsed().as_secs_f64() * 1e3)
                .unwrap_or(0.0);
            println!(
                "[GPU wgpu] readback: copy+sync {:.1}ms  convert {:.1}ms  total {:.1}ms  ({:.0} MB)",
                total_ms - convert_ms,
                convert_ms,
                total_ms,
                (self.k * self.n * 4) as f64 / 1e6,
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

/// Hardware binding limit — queried from the adapter once, without creating a
/// full device, so `fits_in_gpu` can be called cheaply before any GPU state
/// is initialised.
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
        .unwrap_or(128 * 1024 * 1024) // safe fallback if no adapter
    })
}

/// Full GPU state: device, pipelines, and per-dimension solver buffers.
/// Stored in a single `Mutex<Option<…>>` so the device can be replaced
/// atomically when the grid dimensions change.
struct GpuState {
    ctx: GpuContext,
    solver: Option<GpuFimSolver>,
    width: usize,
    height: usize,
}

static GPU_STATE: OnceLock<Mutex<Option<GpuState>>> = OnceLock::new();

/// Returns true when `{k, n}` can be handled by the GPU path.
///
/// The only hard constraint is that u_buf (k×n×4 bytes) must fit within the
/// device's native `max_storage_buffer_binding_size` (at 80 % headroom).
/// There is no longer a separate staging-pool cap: when the grid dimensions
/// change, `with_solver` destroys the entire `wgpu::Device`, which releases
/// the internally pooled SHARED staging memory back to Metal and starts the
/// next geometry with a clean slate.
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
        // Destroy the entire device.  Dropping `GpuContext` releases the
        // `wgpu::Device`, which in turn releases all internally pooled SHARED
        // staging memory back to Metal.  The next geometry therefore starts
        // with a clean SHARED heap regardless of how much was written before.
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
            // Same grid, different k: keep the device (pipeline compilation is
            // expensive) and only replace the dimension-specific buffers.
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
    f(&state.ctx, state.solver.as_mut().unwrap())
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
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.init_u_cold(ctx, sources);
        s.upload_speed_and_sources(ctx, speed_field, sources, cell_size);
        s.reset_tile_round(ctx);
        let slots = s.seed_slots_cold(sources);
        s.arm_seed(ctx, &slots);
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] upload+init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] GPU compute: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        s.readback(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}

/// Warm-start solve: keeps the GPU's u_buf from the previous solve and re-solves
/// only tiles covering `changed_cells`.
///
/// `prior_ptrs` is intentionally ignored — the GPU already holds the prior
/// travel-time field in `u_buf`, avoiding an f32↔f64 roundtrip.
pub fn solve_warm(
    out_ptrs: &[usize],
    _prior_ptrs: &[usize],
    n: usize,
    speed_field: &[f64],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        let t_upload = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.upload_speed_and_sources(ctx, speed_field, sources, cell_size);
        if s.u_buf_valid {
            s.set_source_cells(ctx, sources);
        } else {
            s.init_u_cold(ctx, sources);
        }
        s.reset_tile_round(ctx);
        let slots = if s.u_buf_valid && !changed_cells.is_empty() {
            s.seed_slots_warm(changed_cells, sources)
        } else {
            s.seed_slots_cold(sources)
        };
        s.arm_seed(ctx, &slots);
        if let Some(t) = t_upload {
            println!(
                "[GPU wgpu] upload+init: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_gpu = crate::PRINT_TIMINGS.then(std::time::Instant::now);
        s.encode_and_submit(ctx);
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] GPU compute: {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        s.readback(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}

/// f32-native cold solve: `speed_field` is `f32`, `out_ptrs` point to `*mut f32` buffers.
/// Skips the `f64 → f32` conversion allocation — zero-copy upload from the caller's buffer.
pub fn solve_cold_f32(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f32,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        s.init_u_cold(ctx, sources);
        s.upload_speed_and_sources_f32(ctx, speed_field, sources, cell_size);
        s.reset_tile_round(ctx);
        let slots = s.seed_slots_cold(sources);
        s.arm_seed(ctx, &slots);
        s.encode_and_submit(ctx);
        s.readback_f32(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}

/// f32-native warm solve: `speed_field` is `f32`, `out_ptrs` point to `*mut f32` buffers.
pub fn solve_warm_f32(
    out_ptrs: &[usize],
    n: usize,
    speed_field: &[f32],
    changed_cells: &[u32],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f32,
) {
    let k = out_ptrs.len();
    with_solver(k, n, width, height, |ctx, s| {
        s.upload_speed_and_sources_f32(ctx, speed_field, sources, cell_size);
        if s.u_buf_valid {
            s.set_source_cells(ctx, sources);
        } else {
            s.init_u_cold(ctx, sources);
        }
        s.reset_tile_round(ctx);
        let slots = if s.u_buf_valid && !changed_cells.is_empty() {
            s.seed_slots_warm(changed_cells, sources)
        } else {
            s.seed_slots_cold(sources)
        };
        s.arm_seed(ctx, &slots);
        s.encode_and_submit(ctx);
        s.readback_f32(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}

// ── Multi-source public entry points ─────────────────────────────────────────
//
// These accept CSR-format sources: destination `d` uses
// `sources_flat[src_offsets[d]..src_offsets[d+1]]`.
//
// The GPU shader and active-tile pipeline are unchanged.  All that differs
// from the single-source paths is how source cells are initialised in u_buf
// (multiple writes per dest) and how seed tiles are computed (union over all
// source cells per dest).  The sources_buf that the shader reads holds the
// first source cell per destination — enough for the shader's convergence
// check; FIM can never increase a value, so non-first source cells that
// start at 0 remain at 0 throughout the solve.

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
        s.upload_speed_and_sources(ctx, speed_field, &first, cell_size);
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
        if let Some(t) = t_gpu {
            println!(
                "[GPU wgpu] GPU compute (ms): {:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        s.readback(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}

pub fn solve_warm_ms(
    out_ptrs: &[usize],
    _prior_ptrs: &[usize],
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
        s.upload_speed_and_sources(ctx, speed_field, &first, cell_size);
        if s.u_buf_valid {
            s.set_source_cells_ms(ctx, sources_flat, src_offsets);
        } else {
            s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        }
        s.reset_tile_round(ctx);
        let slots = if s.u_buf_valid && !changed_cells.is_empty() {
            s.seed_slots_warm_ms(changed_cells, sources_flat, src_offsets)
        } else {
            s.seed_slots_cold_ms(sources_flat, src_offsets)
        };
        s.arm_seed(ctx, &slots);
        s.encode_and_submit(ctx);
        s.readback(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
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
    with_solver(k, n, width, height, |ctx, s| {
        let first = first_sources(sources_flat, src_offsets);
        s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        s.upload_speed_and_sources_f32(ctx, speed_field, &first, cell_size);
        s.reset_tile_round(ctx);
        let slots = s.seed_slots_cold_ms(sources_flat, src_offsets);
        s.arm_seed(ctx, &slots);
        s.encode_and_submit(ctx);
        s.readback_f32(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}

pub fn solve_warm_ms_f32(
    out_ptrs: &[usize],
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
        s.upload_speed_and_sources_f32(ctx, speed_field, &first, cell_size);
        if s.u_buf_valid {
            s.set_source_cells_ms(ctx, sources_flat, src_offsets);
        } else {
            s.init_u_cold_ms(ctx, sources_flat, src_offsets);
        }
        s.reset_tile_round(ctx);
        let slots = if s.u_buf_valid && !changed_cells.is_empty() {
            s.seed_slots_warm_ms(changed_cells, sources_flat, src_offsets)
        } else {
            s.seed_slots_cold_ms(sources_flat, src_offsets)
        };
        s.arm_seed(ctx, &slots);
        s.encode_and_submit(ctx);
        s.readback_f32(ctx, out_ptrs);
        s.u_buf_valid = true;
    });
}
