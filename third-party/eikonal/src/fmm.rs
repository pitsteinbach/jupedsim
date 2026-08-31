use num_traits::Float;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
enum State {
    Far,
    Tentative,
    Accepted,
}

/// Wraps a Float to provide total order (NaN compares equal to itself, sorts last).
#[derive(PartialEq, Clone, Copy)]
struct OrdFloat<T: Float + Copy + PartialEq>(T);

impl<T: Float + Copy + PartialEq> Eq for OrdFloat<T> {}

impl<T: Float + Copy + PartialEq> PartialOrd for OrdFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float + Copy + PartialEq> Ord for OrdFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NaN treated as greater than any finite value — consistent total order.
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Solves the eikonal equation |∇u| = 1/f using the Fast Marching Method (Sethian 1996).
/// Complexity O(N log N). f64 concrete entry point — keeps the cxx bridge signatures unchanged.
pub fn solve(
    speed_field: &[f64],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: f64,
) -> Vec<f64> {
    solve_typed(speed_field, sources, width, height, cell_size)
}

/// Allocation-free f64 variant.
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

/// Generic solve over `f32` or `f64`.
pub fn solve_typed<T: Float + Copy + PartialEq>(
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

/// Allocation-free generic variant.
pub fn solve_into_typed<T: Float + Copy + PartialEq>(
    out: &mut [T],
    speed_field: &[T],
    sources: &[u32],
    width: usize,
    height: usize,
    cell_size: T,
) {
    let t_total = std::time::Instant::now();
    let n = width * height;
    out.fill(T::infinity());
    let mut state = vec![State::Far; n];
    let mut heap: BinaryHeap<(Reverse<OrdFloat<T>>, usize)> = BinaryHeap::new();

    let t_init = std::time::Instant::now();
    for &s in sources {
        let idx = s as usize;
        out[idx] = T::zero();
        state[idx] = State::Accepted;
    }
    for &s in sources {
        let idx = s as usize;
        relax_neighbors(idx, out, &mut state, speed_field, width, height, cell_size, &mut heap);
    }
    let dur_init = t_init.elapsed();

    let t_heap = std::time::Instant::now();
    let mut heap_pops = 0usize;
    while let Some((Reverse(OrdFloat(val)), idx)) = heap.pop() {
        heap_pops += 1;
        if state[idx] == State::Accepted || val > out[idx] {
            continue;
        }
        state[idx] = State::Accepted;
        relax_neighbors(idx, out, &mut state, speed_field, width, height, cell_size, &mut heap);
    }
    let dur_heap = t_heap.elapsed();

    if crate::PRINT_TIMINGS {
        println!(
            "[FMM] N={n} heap_pops={heap_pops} | init={:.1}ms heap={:.1}ms total={:.1}ms",
            dur_init.as_secs_f64() * 1e3,
            dur_heap.as_secs_f64() * 1e3,
            t_total.elapsed().as_secs_f64() * 1e3,
        );
    }
}

fn relax_neighbors<T: Float + Copy + PartialEq>(
    origin: usize,
    u: &mut [T],
    state: &mut [State],
    speed_field: &[T],
    width: usize,
    height: usize,
    cell_size: T,
    heap: &mut BinaryHeap<(Reverse<OrdFloat<T>>, usize)>,
) {
    let i = origin / width;
    let j = origin % width;
    for nb in four_neighbors(i, j, width, height) {
        if state[nb] == State::Accepted {
            continue;
        }
        let f = speed_field[nb];
        if f <= T::zero() {
            continue;
        }
        let ni = nb / width;
        let nj = nb % width;
        let a = min_accepted_x(u, state, ni, nj, width);
        let b = min_accepted_y(u, state, ni, nj, width, height);
        let candidate = godunov_update(a, b, cell_size / f);
        if candidate < u[nb] {
            u[nb] = candidate;
            state[nb] = State::Tentative;
            heap.push((Reverse(OrdFloat(candidate)), nb));
        }
    }
}

fn godunov_update<T: Float>(a: T, b: T, cost: T) -> T {
    let lo = a.min(b);
    let hi = a.max(b);
    let u1 = lo + cost;
    if u1 <= hi {
        return u1;
    }
    let two = T::one() + T::one();
    let disc = two * cost * cost - (a - b) * (a - b);
    if disc >= T::zero() { (a + b + disc.sqrt()) / two } else { u1 }
}

fn four_neighbors(i: usize, j: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let mut nb = [0usize; 4];
    let mut count = 0usize;
    if j > 0 { nb[count] = i * width + j - 1; count += 1; }
    if j + 1 < width { nb[count] = i * width + j + 1; count += 1; }
    if i > 0 { nb[count] = (i - 1) * width + j; count += 1; }
    if i + 1 < height { nb[count] = (i + 1) * width + j; count += 1; }
    nb.into_iter().take(count)
}

fn min_accepted_x<T: Float>(u: &[T], state: &[State], i: usize, j: usize, width: usize) -> T {
    let left = if j > 0 && state[i * width + j - 1] == State::Accepted {
        u[i * width + j - 1]
    } else {
        T::infinity()
    };
    let right = if j + 1 < width && state[i * width + j + 1] == State::Accepted {
        u[i * width + j + 1]
    } else {
        T::infinity()
    };
    left.min(right)
}

fn min_accepted_y<T: Float>(
    u: &[T],
    state: &[State],
    i: usize,
    j: usize,
    width: usize,
    height: usize,
) -> T {
    let up = if i > 0 && state[(i - 1) * width + j] == State::Accepted {
        u[(i - 1) * width + j]
    } else {
        T::infinity()
    };
    let down = if i + 1 < height && state[(i + 1) * width + j] == State::Accepted {
        u[(i + 1) * width + j]
    } else {
        T::infinity()
    };
    up.min(down)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_point_source() {
        let w = 101usize;
        let h = 101usize;
        let cx = w / 2;
        let cy = h / 2;
        let speed = vec![1.0_f64; w * h];
        let sources = vec![(cy * w + cx) as u32];
        let u = solve(&speed, &sources, w, h, 1.0);
        let cases = [((cy, cx + 10), 0.02), ((cy + 10, cx), 0.02), ((cy + 10, cx + 10), 0.08)];
        for ((i, j), tol) in cases {
            let expected =
                (((i as f64 - cy as f64).powi(2) + (j as f64 - cx as f64).powi(2)).sqrt());
            let got = u[i * w + j];
            assert!(
                (got - expected).abs() / expected < tol,
                "({i},{j}): expected {expected:.3}, got {got:.3} (tol {tol})"
            );
        }
    }

    #[test]
    fn obstacle_isolation() {
        let w = 5usize;
        let h = 1usize;
        let mut speed = vec![1.0_f64; w * h];
        speed[2] = 0.0;
        let u = solve(&speed, &[0u32], w, h, 1.0);
        assert!(u[3].is_infinite());
    }

    #[test]
    fn multiple_sources() {
        let w = 11usize;
        let speed = vec![1.0_f64; w];
        let u = solve(&speed, &[0u32, 10u32], w, 1, 1.0);
        assert_eq!(u[0], 0.0);
        assert_eq!(u[10], 0.0);
        assert!((u[5] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn f32_smoke() {
        let w = 21usize;
        let speed = vec![1.0_f32; w * w];
        let u = solve_typed::<f32>(&speed, &[((w / 2) * w + w / 2) as u32], w, w, 1.0_f32);
        assert!(u[(w / 2) * w + w / 2] == 0.0);
        assert!(u[0].is_finite());
    }
}
