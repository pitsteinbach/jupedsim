use num_traits::Float;

/// Solves the eikonal equation |∇u| = 1/f on a 2D regular grid using the
/// Fast Sweeping Method (Zhao 2005). Complexity: O(N) where N = width * height.
pub fn solve(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    solve_typed(speed_field, sources, width, height, cell_size)
}

/// Allocation-free variant: initialises `out` to INFINITY and solves in place.
pub fn solve_into(
    out: &mut [f64],
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) {
    solve_into_typed(out, speed_field, sources, width, height, cell_size);
}

/// Generic solve: accepts `f32` or `f64`. Returns a `Vec<T>`.
pub fn solve_typed<T: Float>(
    speed_field: &[T],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: T,
) -> Vec<T> {
    let n = width * height;
    let mut u = vec![T::infinity(); n];
    solve_into_typed(&mut u, speed_field, sources, width, height, cell_size);
    u
}

/// Allocation-free generic variant: initialises `out` to INFINITY and solves in place.
pub fn solve_into_typed<T: Float>(
    out: &mut [T],
    speed_field: &[T],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: T,
) {
    let t = std::time::Instant::now();
    out.fill(T::infinity());
    for &s in sources {
        out[s as usize] = T::zero();
    }
    for _ in 0..2 {
        sweep(out, speed_field, width, height, cell_size, false, false);
        sweep(out, speed_field, width, height, cell_size, false, true);
        sweep(out, speed_field, width, height, cell_size, true, false);
        sweep(out, speed_field, width, height, cell_size, true, true);
    }
    if crate::PRINT_TIMINGS {
        let n = width * height;
        println!("[FSM] N={n} total={:.1}ms", t.elapsed().as_secs_f64() * 1e3);
    }
}

fn sweep<T: Float>(
    u: &mut [T],
    speed: &[T],
    width: usize,
    height: usize,
    h: T,
    i_rev: bool,
    j_rev: bool,
) {
    let rows = iter_range(height, i_rev);
    for i in rows {
        for j in iter_range(width, j_rev) {
            let idx = i * width + j;
            let f = speed[idx];
            if f <= T::zero() {
                continue;
            }
            let cost = h / f;
            let a = min_neighbor_x(u, i, j, width);
            let b = min_neighbor_y(u, i, j, width, height);
            let candidate = godunov_update(a, b, cost);
            if candidate < u[idx] {
                u[idx] = candidate;
            }
        }
    }
}

/// Upwind Godunov update: solves (u-a)₊² + (u-b)₊² = cost²
fn godunov_update<T: Float>(a: T, b: T, cost: T) -> T {
    let lo = a.min(b);
    let hi = a.max(b);

    let u1 = lo + cost;
    if u1 <= hi {
        return u1;
    }

    let two = T::one() + T::one();
    let disc = two * cost * cost - (a - b) * (a - b);
    if disc >= T::zero() {
        (a + b + disc.sqrt()) / two
    } else {
        u1
    }
}

fn min_neighbor_x<T: Float>(u: &[T], i: usize, j: usize, width: usize) -> T {
    let left = if j > 0 { u[i * width + j - 1] } else { T::infinity() };
    let right = if j + 1 < width { u[i * width + j + 1] } else { T::infinity() };
    left.min(right)
}

fn min_neighbor_y<T: Float>(u: &[T], i: usize, j: usize, width: usize, height: usize) -> T {
    let up = if i > 0 { u[(i - 1) * width + j] } else { T::infinity() };
    let down = if i + 1 < height { u[(i + 1) * width + j] } else { T::infinity() };
    up.min(down)
}

enum Indices {
    Fwd(std::ops::Range<usize>),
    Rev(std::iter::Rev<std::ops::Range<usize>>),
}

impl Iterator for Indices {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Indices::Fwd(r) => r.next(),
            Indices::Rev(r) => r.next(),
        }
    }
}

fn iter_range(len: usize, rev: bool) -> Indices {
    if rev { Indices::Rev((0..len).rev()) } else { Indices::Fwd(0..len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_point_source() {
        let w = 101usize;
        let h = 101usize;
        let h_size = 1.0_f64;
        let cx = w / 2;
        let cy = h / 2;
        let speed = vec![1.0_f64; w * h];
        let sources = vec![(cy * w + cx) as u32];
        let u = solve(&speed, &sources, w, h, h_size);
        let cases = [
            ((cy, cx + 10), 0.02),
            ((cy + 10, cx), 0.02),
            ((cy + 10, cx + 10), 0.08),
        ];
        for ((i, j), tol) in cases {
            let expected =
                (((i as f64 - cy as f64).powi(2) + (j as f64 - cx as f64).powi(2)).sqrt()) * h_size;
            let got = u[i * w + j];
            assert!(
                (got - expected).abs() / expected < tol,
                "({i},{j}): expected {expected:.3}, got {got:.3} (tol {tol})"
            );
        }
    }

    #[test]
    fn uniform_point_source_f32() {
        let w = 51usize;
        let h = 51usize;
        let cx = w / 2;
        let cy = h / 2;
        let speed = vec![1.0_f32; w * h];
        let sources = vec![(cy * w + cx) as u32];
        let u = solve_typed::<f32>(&speed, &sources, w, h, 1.0_f32);
        let center_neighbor = u[cy * w + cx + 1];
        assert!(center_neighbor > 0.9 && center_neighbor < 1.1, "got {center_neighbor}");
    }
}
