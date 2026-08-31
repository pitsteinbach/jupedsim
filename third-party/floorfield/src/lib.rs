pub mod geometry;
pub mod mesh;

use std::collections::HashMap;

use geometry::GridParams;
use num_traits::Float;
use rayon::prelude::*;
use tracing::instrument;

// ─── Tracing init ────────────────────────────────────────────────────────────
// Respects RUST_LOG (e.g. `RUST_LOG=floorfield=trace`).  No-op when a
// subscriber is already installed or when the env var is absent.
fn maybe_init_tracing() {
    use std::sync::OnceLock;
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};
        let _ = tracing_subscriber::fmt()
            .with_span_events(FmtSpan::CLOSE)
            .with_env_filter(EnvFilter::from_default_env())
            .try_init();
    });
}

// ─── Solver selector ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum EikonalSolver {
    Fsm,
    Fmm,
    Fim,
}

// ─── Per-type solver dispatch ────────────────────────────────────────────────

trait EikonalOps: Float + Send + Sync + Copy + PartialEq + num_traits::FromPrimitive + 'static {
    fn fsm_solve_into(out: &mut [Self], speed: &[Self], src: &[u32], w: usize, h: usize, cs: Self);
    fn fmm_solve_into(out: &mut [Self], speed: &[Self], src: &[u32], w: usize, h: usize, cs: Self);
    /// FIM single-source (falls back to FSM for f32 — CPU FIM is f64-only).
    fn fim_solve_into(out: &mut [Self], speed: &[Self], src: &[u32], w: usize, h: usize, cs: Self);
    /// Warm single-source FIM (falls back to cold FSM for f32).
    fn fim_warm_into(
        out: &mut [Self],
        speed: &[Self],
        prior: &[Self],
        changed: &[u32],
        src: &[u32],
        w: usize,
        h: usize,
        cs: Self,
    );
    /// Batch cold: GPU FIM (single-source) / CPU FIM (multi-source) / Rayon-FSM fallback.
    /// `src_offsets` is CSR: dest `i` uses `sources_flat[src_offsets[i]..src_offsets[i+1]]`.
    fn fim_batch_cold(
        out_ptrs: &[usize],
        n: usize,
        speed: &[Self],
        sources_flat: &[u32],
        src_offsets: &[u32],
        w: usize,
        h: usize,
        cs: Self,
    );
    /// Batch warm: GPU FIM (single-source) / CPU FIM warm (multi-source) / Rayon fallback.
    fn fim_batch_warm(
        out_ptrs: &[usize],
        prior_ptrs: &[usize],
        n: usize,
        speed: &[Self],
        changed: &[u32],
        sources_flat: &[u32],
        src_offsets: &[u32],
        w: usize,
        h: usize,
        cs: Self,
    );
    fn cast_f64(d: f64) -> Self;
}

impl EikonalOps for f64 {
    fn fsm_solve_into(out: &mut [f64], speed: &[f64], src: &[u32], w: usize, h: usize, cs: f64) {
        eikonal::fsm::solve_into(out, speed, src, w, h, cs);
    }
    fn fmm_solve_into(out: &mut [f64], speed: &[f64], src: &[u32], w: usize, h: usize, cs: f64) {
        eikonal::fmm::solve_into(out, speed, src, w, h, cs);
    }
    fn fim_solve_into(out: &mut [f64], speed: &[f64], src: &[u32], w: usize, h: usize, cs: f64) {
        eikonal::fim::solve_into(out, speed, src, w, h, cs);
    }
    fn fim_warm_into(
        out: &mut [f64],
        speed: &[f64],
        prior: &[f64],
        changed: &[u32],
        src: &[u32],
        w: usize,
        h: usize,
        cs: f64,
    ) {
        eikonal::fim::solve_warm_into(out, speed, prior, changed, src, w, h, cs);
    }
    fn fim_batch_cold(
        out_ptrs: &[usize],
        n: usize,
        speed: &[f64],
        sources_flat: &[u32],
        src_offsets: &[u32],
        w: usize,
        h: usize,
        cs: f64,
    ) {
        eikonal::fim_batch_cold_ms(out_ptrs, n, speed, sources_flat, src_offsets, w, h, cs);
    }
    fn fim_batch_warm(
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
        eikonal::fim_batch_warm_ms(
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
    }
    fn cast_f64(d: f64) -> f64 {
        d
    }
}

impl EikonalOps for f32 {
    fn fsm_solve_into(out: &mut [f32], speed: &[f32], src: &[u32], w: usize, h: usize, cs: f32) {
        eikonal::fsm::solve_into_typed(out, speed, src, w, h, cs);
    }
    fn fmm_solve_into(out: &mut [f32], speed: &[f32], src: &[u32], w: usize, h: usize, cs: f32) {
        eikonal::fmm::solve_into_typed(out, speed, src, w, h, cs);
    }
    fn fim_solve_into(out: &mut [f32], speed: &[f32], src: &[u32], w: usize, h: usize, cs: f32) {
        eikonal::fsm::solve_into_typed(out, speed, src, w, h, cs); // CPU FIM is f64-only
    }
    fn fim_warm_into(
        out: &mut [f32],
        speed: &[f32],
        _prior: &[f32],
        _changed: &[u32],
        src: &[u32],
        w: usize,
        h: usize,
        cs: f32,
    ) {
        eikonal::fsm::solve_into_typed(out, speed, src, w, h, cs); // cold re-solve
    }
    fn fim_batch_cold(
        out_ptrs: &[usize],
        n: usize,
        speed: &[f32],
        sources_flat: &[u32],
        src_offsets: &[u32],
        w: usize,
        h: usize,
        cs: f32,
    ) {
        eikonal::fim_batch_cold_ms_f32(out_ptrs, n, speed, sources_flat, src_offsets, w, h, cs);
    }
    fn fim_batch_warm(
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
        eikonal::fim_batch_warm_ms_f32(
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
    }
    fn cast_f64(d: f64) -> f32 {
        d as f32
    }
}

// ─── Unified destination entry (polygon or point) ────────────────────────────
// Double-buffered so warm restarts are O(1) flips with no data movement.

struct DestEntry<T> {
    sources: Vec<u32>, // 1 cell for point dests, N cells for polygon dests
    bufs: [Vec<T>; 2],
    active: usize,
    has_prior: bool,
    valid: bool,
}

impl<T: Clone> DestEntry<T> {
    fn new(sources: Vec<u32>) -> Self {
        Self {
            sources,
            bufs: [Vec::new(), Vec::new()],
            active: 0,
            has_prior: false,
            valid: false,
        }
    }
    fn current_buf(&self) -> &Vec<T> {
        &self.bufs[self.active]
    }
    fn other_buf_mut(&mut self) -> &mut Vec<T> {
        &mut self.bufs[1 - self.active]
    }
    fn flip(&mut self) {
        self.active ^= 1;
        self.valid = true;
        self.has_prior = true;
    }
    fn mark_stale(&mut self) {
        self.valid = false;
    }
    fn invalidate(&mut self) {
        self.valid = false;
        self.has_prior = false;
        self.active = 0;
    }
}

// ─── CSR helper ─────────────────────────────────────────────────────────────

/// Build a flat sources array and CSR offsets for a slice of destination IDs.
/// `src_offsets[i+1] - src_offsets[i]` is the number of sources for `ids[i]`.
fn build_csr<T>(ids: &[usize], destinations: &[DestEntry<T>]) -> (Vec<u32>, Vec<u32>) {
    let mut sources_flat: Vec<u32> = Vec::new();
    let mut src_offsets: Vec<u32> = Vec::with_capacity(ids.len() + 1);
    src_offsets.push(0);
    for &id in ids {
        sources_flat.extend_from_slice(&destinations[id].sources);
        src_offsets.push(sources_flat.len() as u32);
    }
    (sources_flat, src_offsets)
}

// ─── Generic floor-field core ────────────────────────────────────────────────

struct FloorfieldInner<T: EikonalOps> {
    grid: GridParams,
    speed_field: Vec<T>,
    dynamic_speed_field: Vec<T>,
    prev_dynamic_speed_field: Vec<T>,
    density_field: Vec<T>,
    // Precomputed Gaussian KDE kernel — spreads each agent over a fixed physical
    // footprint (sigma = 0.5 m) so density is independent of cell size.
    kernel_offsets: Vec<(i32, i32)>,
    kernel_weights: Vec<T>, // normalised: sum(w) * cell_area == 1.0 per agent
    destinations: Vec<DestEntry<T>>,
    cell_to_id: HashMap<u32, usize>,
    // destinations[0..n_polygon_dests] are polygon (ID-based) dests; the rest
    // are point dests added on demand and truncated on clear_point_cache.
    n_polygon_dests: usize,
    changed_cells: Vec<u32>,
    step_counter: i32,
    recompute_interval: i32,
    dynamic_field_built: bool,
    solver: EikonalSolver,
    solver_benchmarked: bool,
    ceil_density: T,
    jam_density: T,
    last_travel_times: Vec<T>,
}

impl<T: EikonalOps> FloorfieldInner<T> {
    fn build_kde_kernel(cell_size: f64, sigma: f64) -> (Vec<(i32, i32)>, Vec<T>) {
        let sigma2 = sigma * sigma;
        let r = (3.0 * sigma / cell_size).ceil() as i32;
        let cell_area = cell_size * cell_size;
        let mut offsets = Vec::new();
        let mut raw_w: Vec<f64> = Vec::new();
        for dr in -r..=r {
            for dc in -r..=r {
                let dx = dc as f64 * cell_size;
                let dy = dr as f64 * cell_size;
                let d2 = dx * dx + dy * dy;
                if d2 > 9.0 * sigma2 {
                    continue;
                }
                offsets.push((dr, dc));
                raw_w.push((-0.5 * d2 / sigma2).exp());
            }
        }
        let wsum: f64 = raw_w.iter().sum();
        let norm = wsum * cell_area;
        let weights = raw_w.iter().map(|&w| T::cast_f64(w / norm)).collect();
        (offsets, weights)
    }

    fn new(grid: GridParams, speed_field: Vec<T>) -> Self {
        let n = grid.width as usize * grid.height as usize;
        let dynamic = speed_field.clone();
        let (kernel_offsets, kernel_weights) = Self::build_kde_kernel(grid.cell_size, 0.5);
        let (solver, solver_benchmarked) = match std::env::var("FLOORFIELD_SOLVER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "fmm" => (EikonalSolver::Fmm, true),
            "fim" => (EikonalSolver::Fim, true),
            "fsm" => (EikonalSolver::Fsm, true),
            _ => (EikonalSolver::Fsm, false),
        };
        Self {
            grid,
            speed_field,
            dynamic_speed_field: dynamic,
            prev_dynamic_speed_field: Vec::new(),
            density_field: vec![T::zero(); n],
            kernel_offsets,
            kernel_weights,
            destinations: Vec::new(),
            cell_to_id: HashMap::new(),
            n_polygon_dests: 0,
            changed_cells: Vec::new(),
            step_counter: 0,
            recompute_interval: 200,
            dynamic_field_built: false,
            solver,
            solver_benchmarked,
            ceil_density: T::cast_f64(2.0),
            jam_density: T::cast_f64(6.0),
            last_travel_times: Vec::new(),
        }
    }

    fn cell_size_t(&self) -> T {
        T::from(self.grid.cell_size).expect("cell_size representable as T")
    }

    fn snap_to_cell(&self, px: f64, py: f64) -> u32 {
        let col = ((px - self.grid.origin[0]) / self.grid.cell_size)
            .floor()
            .clamp(0.0, self.grid.width as f64 - 1.0) as u32;
        let row = ((py - self.grid.origin[1]) / self.grid.cell_size)
            .floor()
            .clamp(0.0, self.grid.height as f64 - 1.0) as u32;
        row * self.grid.width + col
    }

    fn is_routable(&self, px: f64, py: f64) -> bool {
        let col = ((px - self.grid.origin[0]) / self.grid.cell_size).floor() as i64;
        let row = ((py - self.grid.origin[1]) / self.grid.cell_size).floor() as i64;
        if row < 0 || row >= self.grid.height as i64 {
            return false;
        }
        if col < 0 || col >= self.grid.width as i64 {
            return false;
        }
        self.speed_field[row as usize * self.grid.width as usize + col as usize] > T::zero()
    }

    fn add_destination_cells(&mut self, cells: &[u32]) -> usize {
        let mut sorted = cells.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.is_empty() {
            sorted.push(0);
        }
        let id = self.destinations.len();
        self.destinations.push(DestEntry::new(sorted));
        self.n_polygon_dests = self.destinations.len();
        id
    }

    fn add_destination(&mut self, outer: &[f64], holes: &[f64], hole_lengths: &[u32]) -> usize {
        use geo::{Contains, Coord, LineString, Point, Polygon};

        let exterior: Vec<Coord<f64>> = outer
            .chunks_exact(2)
            .map(|c| Coord { x: c[0], y: c[1] })
            .collect();
        let mut off = 0usize;
        let hole_rings: Vec<LineString<f64>> = hole_lengths
            .iter()
            .map(|&len| {
                let end = off + len as usize;
                let ring = LineString::new(
                    holes[off..end]
                        .chunks_exact(2)
                        .map(|c| Coord { x: c[0], y: c[1] })
                        .collect::<Vec<_>>(),
                );
                off = end;
                ring
            })
            .collect();
        // `off` captured mutably inside map closures — use a loop instead:
        let mut off = 0usize;
        let mut hole_rings_built = Vec::with_capacity(hole_lengths.len());
        for &len in hole_lengths {
            let end = off + len as usize;
            hole_rings_built.push(LineString::new(
                holes[off..end]
                    .chunks_exact(2)
                    .map(|c| Coord { x: c[0], y: c[1] })
                    .collect::<Vec<_>>(),
            ));
            off = end;
        }
        let _ = hole_rings; // discard iterator version; hole_rings_built is correct

        let polygon = Polygon::new(LineString::new(exterior), hole_rings_built);
        let mut sources: Vec<u32> = Vec::new();
        for row in 0..self.grid.height {
            for col in 0..self.grid.width {
                let px = self.grid.origin[0] + (col as f64 + 0.5) * self.grid.cell_size;
                let py = self.grid.origin[1] + (row as f64 + 0.5) * self.grid.cell_size;
                if polygon.contains(&Point::new(px, py)) {
                    sources.push(row * self.grid.width + col);
                }
            }
        }
        sources.sort_unstable();
        sources.dedup();

        if sources.is_empty() {
            let pts: Vec<Coord<f64>> = outer
                .chunks_exact(2)
                .map(|c| Coord { x: c[0], y: c[1] })
                .collect();
            let (sx, sy) = pts
                .iter()
                .fold((0.0f64, 0.0f64), |acc, p| (acc.0 + p.x, acc.1 + p.y));
            sources.push(self.snap_to_cell(sx / pts.len() as f64, sy / pts.len() as f64));
        }

        let id = self.destinations.len();
        self.destinations.push(DestEntry::new(sources));
        self.n_polygon_dests = self.destinations.len();
        id
    }

    #[instrument(skip_all, fields(n_agents = positions_xy.len() / 2))]
    fn update_density(&mut self, positions_xy: &[f64]) {
        self.step_counter += 1;
        if self.step_counter < self.recompute_interval {
            return;
        }
        self.step_counter = 0;

        self.density_field.fill(T::zero());
        let iw = self.grid.width as i32;
        let ih = self.grid.height as i32;
        let gw = self.grid.width as usize;
        for chunk in positions_xy.chunks_exact(2) {
            let col0 = ((chunk[0] - self.grid.origin[0]) / self.grid.cell_size).floor() as i32;
            let row0 = ((chunk[1] - self.grid.origin[1]) / self.grid.cell_size).floor() as i32;
            if row0 < 0 || row0 >= ih || col0 < 0 || col0 >= iw {
                continue;
            }
            for k in 0..self.kernel_offsets.len() {
                let (dr, dc) = self.kernel_offsets[k];
                let r = row0 + dr;
                let c = col0 + dc;
                if r < 0 || r >= ih || c < 0 || c >= iw {
                    continue;
                }
                let idx = r as usize * gw + c as usize;
                self.density_field[idx] = self.density_field[idx] + self.kernel_weights[k];
            }
        }

        let first_build = !self.dynamic_field_built;
        self.build_dynamic_speed_field();
        if first_build {
            for e in &mut self.destinations {
                e.invalidate();
            }
        } else {
            for e in &mut self.destinations {
                e.mark_stale();
            }
        }
    }

    #[instrument(skip_all)]
    fn build_dynamic_speed_field(&mut self) {
        self.changed_cells.clear();
        let n = self.grid.width as usize * self.grid.height as usize;
        std::mem::swap(
            &mut self.dynamic_speed_field,
            &mut self.prev_dynamic_speed_field,
        );
        self.dynamic_speed_field.resize(n, T::zero());

        let ceil: f64 = num_traits::cast(self.ceil_density).unwrap_or(2.0);
        let jam: f64 = num_traits::cast(self.jam_density).unwrap_or(6.0);

        for idx in 0..n {
            if self.speed_field[idx] <= T::zero() {
                continue;
            }
            let density: f64 = num_traits::cast(self.density_field[idx]).unwrap_or(0.0);
            let modifier = ((1.0 - (density - ceil) / (jam - ceil)).max(0.2)).min(1.0);
            let base: f64 = num_traits::cast(self.speed_field[idx]).unwrap_or(0.0);
            self.dynamic_speed_field[idx] = T::cast_f64(base * modifier);
        }

        if self.dynamic_field_built && self.prev_dynamic_speed_field.len() == n {
            for i in 0..n {
                let nv: f64 = num_traits::cast(self.dynamic_speed_field[i]).unwrap_or(0.0);
                let ov: f64 = num_traits::cast(self.prev_dynamic_speed_field[i]).unwrap_or(0.0);
                if (nv - ov).abs() > 1e-12 {
                    self.changed_cells.push(i as u32);
                }
            }
        }
        self.dynamic_field_built = true;
    }

    fn active_speed(&self) -> &[T] {
        if self.dynamic_speed_field.is_empty() {
            &self.speed_field
        } else {
            &self.dynamic_speed_field
        }
    }

    fn compute_single_floorfield(&self, cell: u32, out: &mut Vec<T>, solver: EikonalSolver) {
        let n = self.grid.width as usize * self.grid.height as usize;
        out.resize(n, T::infinity());
        let sp = self.active_speed();
        let cs = self.cell_size_t();
        let w = self.grid.width as usize;
        let h = self.grid.height as usize;
        match solver {
            EikonalSolver::Fsm => T::fsm_solve_into(out, sp, &[cell], w, h, cs),
            EikonalSolver::Fmm => T::fmm_solve_into(out, sp, &[cell], w, h, cs),
            EikonalSolver::Fim => T::fim_solve_into(out, sp, &[cell], w, h, cs),
        }
    }

    fn get_or_register_point(&mut self, cell: u32) -> usize {
        if let Some(&id) = self.cell_to_id.get(&cell) {
            return id;
        }
        let id = self.destinations.len();
        self.destinations.push(DestEntry::new(vec![cell]));
        self.cell_to_id.insert(cell, id);
        id
    }

    // ── Batch solve helpers ───────────────────────────────────────────────────
    // Work for any mix of single- and multi-source destinations.
    //
    // Cold: FSM/FMM run in rayon parallel (multi-source aware). FIM GPU batch
    //   handles single-source dests; multi-source fall back to per-dest CPU FIM.
    // Warm: FIM GPU batch for single-source; per-dest CPU FIM warm for multi-source.
    //   Warm restarts are always FIM-based — FSM/FMM have no incremental update.
    //
    // Raw-pointer aliasing is safe: we hold &mut self, write only to other_buf
    // (never current_buf), and each id is distinct.

    fn solve_cold_batch(&mut self, ids: &[usize]) {
        if ids.is_empty() {
            return;
        }
        let n = self.grid.width as usize * self.grid.height as usize;
        let cs = self.cell_size_t();
        let w = self.grid.width as usize;
        let h = self.grid.height as usize;
        let speed = self.active_speed().to_vec();
        let solver = self.solver;

        for &id in ids {
            let e = &mut self.destinations[id];
            if e.other_buf_mut().len() != n {
                e.other_buf_mut().resize(n, T::infinity());
            }
        }
        let out_ptrs: Vec<usize> = ids
            .iter()
            .map(|&id| {
                let e = &self.destinations[id];
                e.bufs[1 - e.active].as_ptr() as usize
            })
            .collect();

        if solver == EikonalSolver::Fim {
            // Build CSR; eikonal handles the single/multi-source split internally.
            let (sources_flat, src_offsets) = build_csr(ids, &self.destinations);
            T::fim_batch_cold(&out_ptrs, n, &speed, &sources_flat, &src_offsets, w, h, cs);
        } else {
            // FSM / FMM accept multi-source slices directly; rayon-parallel over dests.
            let do_fmm = solver == EikonalSolver::Fmm;
            let work: Vec<(usize, Vec<u32>)> = ids
                .iter()
                .map(|&id| {
                    let e = &self.destinations[id];
                    (e.bufs[1 - e.active].as_ptr() as usize, e.sources.clone())
                })
                .collect();
            work.par_iter().for_each(|(raw, srcs)| {
                let out = unsafe { std::slice::from_raw_parts_mut(*raw as *mut T, n) };
                if do_fmm {
                    T::fmm_solve_into(out, &speed, srcs, w, h, cs);
                } else {
                    T::fsm_solve_into(out, &speed, srcs, w, h, cs);
                }
            });
        }
        for &id in ids {
            self.destinations[id].flip();
        }
    }

    fn solve_warm_batch(&mut self, ids: &[usize]) {
        if ids.is_empty() {
            return;
        }
        let n = self.grid.width as usize * self.grid.height as usize;
        let cs = self.cell_size_t();
        let w = self.grid.width as usize;
        let h = self.grid.height as usize;
        let speed = self.active_speed().to_vec();
        let changed = self.changed_cells.clone();

        for &id in ids {
            let e = &mut self.destinations[id];
            if e.other_buf_mut().len() != n {
                e.other_buf_mut().resize(n, T::infinity());
            }
        }
        let out_ptrs: Vec<usize> = ids
            .iter()
            .map(|&id| {
                let e = &self.destinations[id];
                e.bufs[1 - e.active].as_ptr() as usize
            })
            .collect();
        let prior_ptrs: Vec<usize> = ids
            .iter()
            .map(|&id| self.destinations[id].current_buf().as_ptr() as usize)
            .collect();
        let (sources_flat, src_offsets) = build_csr(ids, &self.destinations);
        T::fim_batch_warm(
            &out_ptrs,
            &prior_ptrs,
            n,
            &speed,
            &changed,
            &sources_flat,
            &src_offsets,
            w,
            h,
            cs,
        );
        for &id in ids {
            self.destinations[id].flip();
        }
    }

    // ── Unified ensure ────────────────────────────────────────────────────────

    #[instrument(skip_all, fields(dest_id))]
    fn ensure_dest(&mut self, dest_id: usize) {
        if self.destinations[dest_id].valid {
            return;
        }
        if self.destinations[dest_id].has_prior
            && self.dynamic_field_built
            && self.changed_cells.is_empty()
        {
            self.destinations[dest_id].valid = true;
            return;
        }
        let has_prior = self.destinations[dest_id].has_prior;
        let use_warm = has_prior && self.dynamic_field_built && !self.changed_cells.is_empty();

        if !use_warm && !self.solver_benchmarked {
            let src = self.destinations[dest_id].sources[0];
            self.benchmark_solvers(src);
        }
        if use_warm {
            println!("warm restart for dest {}", dest_id);
            self.solve_warm_batch(&[dest_id]);
        } else {
            println!("cold solve for dest {}", dest_id);
            self.solve_cold_batch(&[dest_id]);
        }
        let tt = self.destinations[dest_id].current_buf().clone();
        self.last_travel_times = tt;
    }

    #[instrument(skip_all, fields(n_points = points_xy.len() / 2))]
    fn precompute_destinations(&mut self, points_xy: &[f64]) {
        if self.step_counter != 0 {
            return;
        }

        let mut seen = std::collections::HashSet::new();
        let mut warm_ids: Vec<usize> = Vec::new();
        let mut cold_ids: Vec<usize> = Vec::new();
        let mut unchanged_ids: Vec<usize> = Vec::new();

        for chunk in points_xy.chunks_exact(2) {
            let cell = self.snap_to_cell(chunk[0], chunk[1]);
            let id = self.get_or_register_point(cell);
            if !seen.insert(id) {
                continue;
            }
            let entry = &self.destinations[id];
            if entry.valid {
                continue;
            }
            if entry.has_prior && self.dynamic_field_built {
                if self.changed_cells.is_empty() {
                    unchanged_ids.push(id);
                } else {
                    warm_ids.push(id);
                    //cold_ids.push(id);
                } // warm restart, but also cold solve for benchmarking
            } else {
                cold_ids.push(id);
            }
        }

        if warm_ids.is_empty() && cold_ids.is_empty() && unchanged_ids.is_empty() {
            return;
        }
        if !self.solver_benchmarked && !cold_ids.is_empty() {
            let src = self.destinations[cold_ids[0]].sources[0];
            self.benchmark_solvers(src);
        }

        for &id in &unchanged_ids {
            self.destinations[id].valid = true;
        }

        let _w = (!warm_ids.is_empty())
            .then(|| tracing::trace_span!("warm_batch", n = warm_ids.len()).entered());
        self.solve_warm_batch(&warm_ids);

        let _c = (!cold_ids.is_empty())
            .then(|| tracing::trace_span!("cold_batch", n = cold_ids.len()).entered());
        self.solve_cold_batch(&cold_ids);
    }

    fn compute_gradient(&self, row: i32, col: i32, tt: &[T]) -> (f64, f64) {
        let w = self.grid.width as i32;
        let h = self.grid.height as i32;
        let tc: f64 = num_traits::cast(tt[row as usize * self.grid.width as usize + col as usize])
            .unwrap_or(0.0);
        let get = |r: i32, c: i32| -> f64 {
            if r < 0 || r >= h || c < 0 || c >= w {
                return tc;
            }
            let v: f64 = num_traits::cast(tt[r as usize * self.grid.width as usize + c as usize])
                .unwrap_or(f64::INFINITY);
            if v.is_infinite() {
                tc
            } else {
                v
            }
        };
        let gx = (get(row - 1, col + 1) + 2. * get(row, col + 1) + get(row + 1, col + 1))
            - (get(row - 1, col - 1) + 2. * get(row, col - 1) + get(row + 1, col - 1));
        let gy = (get(row + 1, col - 1) + 2. * get(row + 1, col) + get(row + 1, col + 1))
            - (get(row - 1, col - 1) + 2. * get(row - 1, col) + get(row - 1, col + 1));
        (
            gx / (8.0 * self.grid.cell_size),
            gy / (8.0 * self.grid.cell_size),
        )
    }

    fn gradient_descent_waypoint(&self, px: f64, py: f64, tt: &[T]) -> (f64, f64) {
        let col = ((px - self.grid.origin[0]) / self.grid.cell_size)
            .floor()
            .clamp(0.0, self.grid.width as f64 - 1.0) as i32;
        let row = ((py - self.grid.origin[1]) / self.grid.cell_size)
            .floor()
            .clamp(0.0, self.grid.height as f64 - 1.0) as i32;
        let (gx, gy) = self.compute_gradient(row, col, tt);
        let norm = (gx * gx + gy * gy).sqrt();
        if norm < 1e-12 {
            return (px, py);
        }
        (
            px - self.grid.cell_size * gx / norm,
            py - self.grid.cell_size * gy / norm,
        )
    }

    fn gradient_descent_all_waypoints(
        &self,
        px: f64,
        py: f64,
        dest_x: f64,
        dest_y: f64,
        tt: &[T],
    ) -> Vec<(f64, f64)> {
        let mut path = vec![(px, py)];
        let (mut cx, mut cy) = (px, py);
        let max_steps = 4 * (self.grid.width as i32 + self.grid.height as i32);
        for _ in 0..max_steps {
            let col = ((cx - self.grid.origin[0]) / self.grid.cell_size)
                .floor()
                .clamp(0.0, self.grid.width as f64 - 1.0) as i32;
            let row = ((cy - self.grid.origin[1]) / self.grid.cell_size)
                .floor()
                .clamp(0.0, self.grid.height as f64 - 1.0) as i32;
            let t_val: f64 =
                num_traits::cast(tt[row as usize * self.grid.width as usize + col as usize])
                    .unwrap_or(f64::INFINITY);
            if t_val < self.grid.cell_size * 0.5 {
                break;
            }
            let (gx, gy) = self.compute_gradient(row, col, tt);
            let norm = (gx * gx + gy * gy).sqrt();
            if norm < 1e-12 {
                break;
            }
            cx -= self.grid.cell_size * gx / norm;
            cy -= self.grid.cell_size * gy / norm;
            path.push((cx, cy));
        }
        path.push((dest_x, dest_y));
        path
    }

    fn compute_waypoint_dest(&mut self, px: f64, py: f64, dest_id: usize) -> (f64, f64) {
        self.ensure_dest(dest_id);
        let this = &*self;
        this.gradient_descent_waypoint(px, py, this.destinations[dest_id].current_buf())
    }

    fn compute_all_waypoints_dest(&mut self, px: f64, py: f64, dest_id: usize) -> Vec<(f64, f64)> {
        self.ensure_dest(dest_id);
        let (dest_x, dest_y) = {
            let sources = &self.destinations[dest_id].sources;
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            for &idx in sources {
                sx += self.grid.origin[0]
                    + (idx % self.grid.width) as f64 * self.grid.cell_size
                    + self.grid.cell_size * 0.5;
                sy += self.grid.origin[1]
                    + (idx / self.grid.width) as f64 * self.grid.cell_size
                    + self.grid.cell_size * 0.5;
            }
            let nn = sources.len() as f64;
            (sx / nn, sy / nn)
        };
        let this = &*self;
        this.gradient_descent_all_waypoints(
            px,
            py,
            dest_x,
            dest_y,
            this.destinations[dest_id].current_buf(),
        )
    }

    fn compute_waypoint_point(&mut self, px: f64, py: f64, dest_x: f64, dest_y: f64) -> (f64, f64) {
        let cell = self.snap_to_cell(dest_x, dest_y);
        let id = self.get_or_register_point(cell);
        self.ensure_dest(id);
        let this = &*self;
        this.gradient_descent_waypoint(px, py, this.destinations[id].current_buf())
    }

    fn compute_all_waypoints_point(
        &mut self,
        px: f64,
        py: f64,
        dest_x: f64,
        dest_y: f64,
    ) -> Vec<(f64, f64)> {
        let cell = self.snap_to_cell(dest_x, dest_y);
        let id = self.get_or_register_point(cell);
        self.ensure_dest(id);
        let this = &*self;
        this.gradient_descent_all_waypoints(
            px,
            py,
            dest_x,
            dest_y,
            this.destinations[id].current_buf(),
        )
    }

    fn speed_field_f64(&self) -> Vec<f64> {
        self.speed_field
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(0.0))
            .collect()
    }
    fn travel_times_f64(&self) -> Vec<f64> {
        self.last_travel_times
            .iter()
            .map(|&v| num_traits::cast::<T, f64>(v).unwrap_or(f64::INFINITY))
            .collect()
    }
    fn density_field_f64(&self) -> Vec<f64> {
        self.density_field
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(0.0))
            .collect()
    }
    fn dynamic_speed_field_f64(&self) -> Vec<f64> {
        self.dynamic_speed_field
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(0.0))
            .collect()
    }
    fn set_solver_inner(&mut self, solver: u8) {
        self.solver = match solver {
            1 => EikonalSolver::Fmm,
            2 => EikonalSolver::Fim,
            _ => EikonalSolver::Fsm,
        };
        self.solver_benchmarked = true;
    }
    fn clear_point_cache_inner(&mut self) {
        self.destinations.truncate(self.n_polygon_dests);
        self.cell_to_id.clear();
        self.step_counter = 0;
    }
    fn set_recompute_interval_inner(&mut self, steps: i32) {
        self.recompute_interval = steps;
    }

    fn benchmark_solvers(&mut self, example_cell: u32) {
        const REPS: usize = 5;
        let time = |inner: &FloorfieldInner<T>, s: EikonalSolver| -> f64 {
            let mut buf = Vec::new();
            inner.compute_single_floorfield(example_cell, &mut buf, s);
            let t0 = std::time::Instant::now();
            for _ in 0..REPS {
                inner.compute_single_floorfield(example_cell, &mut buf, s);
            }
            t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
        };
        let (t_fsm, t_fmm, t_fim) = (
            time(self, EikonalSolver::Fsm),
            time(self, EikonalSolver::Fmm),
            time(self, EikonalSolver::Fim),
        );
        eprintln!(
            "[floorfield] benchmark_solvers FSM={t_fsm:.1}ms FMM={t_fmm:.1}ms FIM={t_fim:.1}ms"
        );
        self.solver = if t_fmm <= t_fsm && t_fmm <= t_fim {
            EikonalSolver::Fmm
        } else if t_fim <= t_fsm {
            EikonalSolver::Fim
        } else {
            EikonalSolver::Fsm
        };
        self.solver_benchmarked = true;
    }
}

// ─── Public enum dispatching over precision ──────────────────────────────────

#[allow(private_interfaces)]
pub enum Floorfield {
    F32(FloorfieldInner<f32>),
    F64(FloorfieldInner<f64>),
}

macro_rules! dispatch {
    ($self:expr, $inner:ident => $body:expr) => {
        match $self {
            Floorfield::F32($inner) => $body,
            Floorfield::F64($inner) => $body,
        }
    };
}

// These are the methods cxx calls via self: syntax — signatures must match the bridge exactly.
impl Floorfield {
    fn add_destination_cells(&mut self, cells: &[u32]) -> usize {
        dispatch!(self, inner => inner.add_destination_cells(cells))
    }
    fn add_destination(&mut self, outer: &[f64], holes: &[f64], hole_lengths: &[u32]) -> usize {
        dispatch!(self, inner => inner.add_destination(outer, holes, hole_lengths))
    }
    fn update_density(&mut self, positions_xy: &[f64]) {
        dispatch!(self, inner => inner.update_density(positions_xy))
    }
    fn precompute_destinations(&mut self, points_xy: &[f64]) {
        dispatch!(self, inner => inner.precompute_destinations(points_xy))
    }
    fn is_routable(&self, px: f64, py: f64) -> bool {
        match self {
            Floorfield::F32(inner) => inner.is_routable(px, py),
            Floorfield::F64(inner) => inner.is_routable(px, py),
        }
    }
    fn compute_waypoint_dest(&mut self, px: f64, py: f64, dest_id: usize) -> ffi::Point2d {
        let (x, y) = dispatch!(self, inner => inner.compute_waypoint_dest(px, py, dest_id));
        ffi::Point2d { x, y }
    }
    fn compute_all_waypoints_dest(
        &mut self,
        px: f64,
        py: f64,
        dest_id: usize,
    ) -> Vec<ffi::Point2d> {
        dispatch!(self, inner => inner.compute_all_waypoints_dest(px, py, dest_id))
            .into_iter()
            .map(|(x, y)| ffi::Point2d { x, y })
            .collect()
    }
    fn compute_waypoint_point(
        &mut self,
        px: f64,
        py: f64,
        dest_x: f64,
        dest_y: f64,
    ) -> ffi::Point2d {
        let (x, y) = dispatch!(self, inner => inner.compute_waypoint_point(px, py, dest_x, dest_y));
        ffi::Point2d { x, y }
    }
    fn compute_all_waypoints_point(
        &mut self,
        px: f64,
        py: f64,
        dest_x: f64,
        dest_y: f64,
    ) -> Vec<ffi::Point2d> {
        dispatch!(self, inner => inner.compute_all_waypoints_point(px, py, dest_x, dest_y))
            .into_iter()
            .map(|(x, y)| ffi::Point2d { x, y })
            .collect()
    }
    fn get_speed_field(&self) -> Vec<f64> {
        match self {
            Floorfield::F32(inner) => inner.speed_field_f64(),
            Floorfield::F64(inner) => inner.speed_field_f64(),
        }
    }
    fn get_travel_times(&self) -> Vec<f64> {
        match self {
            Floorfield::F32(inner) => inner.travel_times_f64(),
            Floorfield::F64(inner) => inner.travel_times_f64(),
        }
    }
    fn get_density_field(&self) -> Vec<f64> {
        match self {
            Floorfield::F32(inner) => inner.density_field_f64(),
            Floorfield::F64(inner) => inner.density_field_f64(),
        }
    }
    fn get_dynamic_speed_field(&self) -> Vec<f64> {
        match self {
            Floorfield::F32(inner) => inner.dynamic_speed_field_f64(),
            Floorfield::F64(inner) => inner.dynamic_speed_field_f64(),
        }
    }
    fn set_solver(&mut self, solver: u8) {
        dispatch!(self, inner => inner.set_solver_inner(solver))
    }
    fn clear_point_cache(&mut self) {
        dispatch!(self, inner => inner.clear_point_cache_inner())
    }
    fn set_recompute_interval(&mut self, steps: i32) {
        dispatch!(self, inner => inner.set_recompute_interval_inner(steps))
    }
    fn grid_width(&self) -> u32 {
        match self {
            Floorfield::F32(inner) => inner.grid.width,
            Floorfield::F64(inner) => inner.grid.width,
        }
    }
    fn grid_height(&self) -> u32 {
        match self {
            Floorfield::F32(inner) => inner.grid.height,
            Floorfield::F64(inner) => inner.grid.height,
        }
    }
    fn grid_origin_x(&self) -> f64 {
        match self {
            Floorfield::F32(inner) => inner.grid.origin[0],
            Floorfield::F64(inner) => inner.grid.origin[0],
        }
    }
    fn grid_origin_y(&self) -> f64 {
        match self {
            Floorfield::F32(inner) => inner.grid.origin[1],
            Floorfield::F64(inner) => inner.grid.origin[1],
        }
    }
    fn grid_cell_size(&self) -> f64 {
        match self {
            Floorfield::F32(inner) => inner.grid.cell_size,
            Floorfield::F64(inner) => inner.grid.cell_size,
        }
    }
}

// ─── Constructors (bridge free functions) ────────────────────────────────────

pub fn new_floorfield_f64_from_polygon(
    outer: &[f64],
    holes: &[f64],
    hole_lengths: &[u32],
    cell_size: f64,
    wall_influence_radius: f64,
) -> Box<Floorfield> {
    maybe_init_tracing();
    let (grid, speed) =
        geometry::build_from_polygon(outer, holes, hole_lengths, cell_size, wall_influence_radius);
    Box::new(Floorfield::F64(FloorfieldInner::new(grid, speed)))
}

pub fn new_floorfield_f32_from_polygon(
    outer: &[f64],
    holes: &[f64],
    hole_lengths: &[u32],
    cell_size: f64,
    wall_influence_radius: f64,
) -> Box<Floorfield> {
    maybe_init_tracing();
    let (grid, speed_f64) =
        geometry::build_from_polygon(outer, holes, hole_lengths, cell_size, wall_influence_radius);
    Box::new(Floorfield::F32(FloorfieldInner::new(
        grid,
        speed_f64.iter().map(|&v| v as f32).collect(),
    )))
}

pub fn new_floorfield_f64_from_mesh(
    vertices: &[f64],
    triangles: &[u32],
    walkable: &[u8],
    cell_size: f64,
    wall_influence_radius: f64,
) -> Box<Floorfield> {
    maybe_init_tracing();
    let (grid, speed) = mesh::build_from_mesh(
        vertices,
        triangles,
        walkable,
        cell_size,
        wall_influence_radius,
    );
    Box::new(Floorfield::F64(FloorfieldInner::new(grid, speed)))
}

pub fn new_floorfield_f32_from_mesh(
    vertices: &[f32],
    triangles: &[u32],
    walkable: &[u8],
    cell_size: f64,
    wall_influence_radius: f64,
) -> Box<Floorfield> {
    maybe_init_tracing();
    let (grid, speed_f64) = mesh::build_from_mesh(
        vertices,
        triangles,
        walkable,
        cell_size,
        wall_influence_radius,
    );
    Box::new(Floorfield::F32(FloorfieldInner::new(
        grid,
        speed_f64.iter().map(|&v| v as f32).collect(),
    )))
}

// ─── cxx bridge ─────────────────────────────────────────────────────────────

#[cxx::bridge(namespace = "jupedsim::floorfield")]
mod ffi {
    /// 2-D point passed between Rust and C++.
    #[derive(Clone, Copy)]
    struct Point2d {
        x: f64,
        y: f64,
    }

    extern "Rust" {
        type Floorfield;

        // Constructors — free functions that return an opaque Box<Floorfield>.
        fn new_floorfield_f64_from_polygon(
            outer: &[f64],
            holes: &[f64],
            hole_lengths: &[u32],
            cell_size: f64,
            wall_influence_radius: f64,
        ) -> Box<Floorfield>;
        fn new_floorfield_f32_from_polygon(
            outer: &[f64],
            holes: &[f64],
            hole_lengths: &[u32],
            cell_size: f64,
            wall_influence_radius: f64,
        ) -> Box<Floorfield>;
        fn new_floorfield_f64_from_mesh(
            vertices: &[f64],
            triangles: &[u32],
            walkable: &[u8],
            cell_size: f64,
            wall_influence_radius: f64,
        ) -> Box<Floorfield>;
        fn new_floorfield_f32_from_mesh(
            vertices: &[f32],
            triangles: &[u32],
            walkable: &[u8],
            cell_size: f64,
            wall_influence_radius: f64,
        ) -> Box<Floorfield>;

        // Router-equivalent methods — cxx calls self.method(...) on the Rust type.
        /// Register pre-computed source cells (row*width+col) as a single destination.
        fn add_destination_cells(self: &mut Floorfield, cells: &[u32]) -> usize;
        fn add_destination(
            self: &mut Floorfield,
            outer: &[f64],
            holes: &[f64],
            hole_lengths: &[u32],
        ) -> usize;
        fn update_density(self: &mut Floorfield, positions_xy: &[f64]);
        fn precompute_destinations(self: &mut Floorfield, points_xy: &[f64]);
        fn is_routable(self: &Floorfield, px: f64, py: f64) -> bool;
        fn compute_waypoint_dest(
            self: &mut Floorfield,
            px: f64,
            py: f64,
            dest_id: usize,
        ) -> Point2d;
        fn compute_all_waypoints_dest(
            self: &mut Floorfield,
            px: f64,
            py: f64,
            dest_id: usize,
        ) -> Vec<Point2d>;
        fn compute_waypoint_point(
            self: &mut Floorfield,
            px: f64,
            py: f64,
            dest_x: f64,
            dest_y: f64,
        ) -> Point2d;
        fn compute_all_waypoints_point(
            self: &mut Floorfield,
            px: f64,
            py: f64,
            dest_x: f64,
            dest_y: f64,
        ) -> Vec<Point2d>;

        // Inspection / visualisation accessors.
        fn get_speed_field(self: &Floorfield) -> Vec<f64>;
        fn get_travel_times(self: &Floorfield) -> Vec<f64>;
        fn get_density_field(self: &Floorfield) -> Vec<f64>;
        fn get_dynamic_speed_field(self: &Floorfield) -> Vec<f64>;
        fn set_solver(self: &mut Floorfield, solver: u8);
        fn clear_point_cache(self: &mut Floorfield);
        fn set_recompute_interval(self: &mut Floorfield, steps: i32);
        fn grid_width(self: &Floorfield) -> u32;
        fn grid_height(self: &Floorfield) -> u32;
        fn grid_origin_x(self: &Floorfield) -> f64;
        fn grid_origin_y(self: &Floorfield) -> f64;
        fn grid_cell_size(self: &Floorfield) -> f64;
    }
}
