use geo::{Contains, Coord, LineString, Point, Polygon};

use crate::geometry::GridParams;

/// Build a speed field grid from a triangulated walkable mesh.
///
/// `vertices`: x,y interleaved vertex coordinates (length = 2 × vertex_count).
/// `triangles`: vertex indices, 3 per triangle (length = 3 × triangle_count).
/// `walkable`: 1 per triangle; 1 = walkable, 0 = obstacle (length = triangle_count).
/// `cell_size`: physical size of one grid cell.
/// `wall_influence_radius`: distance from wall boundary at which speed starts being reduced.
///
/// Returns grid parameters and a row-major speed field where 0.0 = obstacle
/// and (0, 1] = walkable, linearly reduced near walls up to `wall_influence_radius`.
pub fn build_from_mesh<T>(
    vertices: &[T],
    triangles: &[u32],
    walkable: &[u8],
    cell_size: f64,
    wall_influence_radius: f64,
) -> (GridParams, Vec<f64>)
where
    T: Copy + Into<f64>,
{
    assert!(vertices.len() % 2 == 0, "vertices must be x,y interleaved");
    assert!(triangles.len() % 3 == 0, "triangles must be 3 indices each");
    assert_eq!(
        walkable.len(),
        triangles.len() / 3,
        "walkable must have one entry per triangle"
    );

    let verts_f64: Vec<[f64; 2]> = vertices
        .chunks_exact(2)
        .map(|c| [c[0].into(), c[1].into()])
        .collect();

    let tri_count = triangles.len() / 3;

    // Compute bounding box over all vertices that appear in walkable triangles.
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for t in 0..tri_count {
        if walkable[t] == 0 {
            continue;
        }
        for k in 0..3 {
            let v = &verts_f64[triangles[t * 3 + k] as usize];
            min_x = min_x.min(v[0]);
            min_y = min_y.min(v[1]);
            max_x = max_x.max(v[0]);
            max_y = max_y.max(v[1]);
        }
    }

    let origin = [min_x, min_y];
    let width = ((max_x - min_x) / cell_size).ceil() as u32;
    let height = ((max_y - min_y) / cell_size).ceil() as u32;

    // Build geo::Polygon for each walkable triangle for point-in-polygon queries.
    let walkable_polys: Vec<Polygon<f64>> = (0..tri_count)
        .filter(|&t| walkable[t] != 0)
        .map(|t| {
            let a = verts_f64[triangles[t * 3] as usize];
            let b = verts_f64[triangles[t * 3 + 1] as usize];
            let c = verts_f64[triangles[t * 3 + 2] as usize];
            Polygon::new(
                LineString::new(vec![
                    Coord { x: a[0], y: a[1] },
                    Coord { x: b[0], y: b[1] },
                    Coord { x: c[0], y: c[1] },
                    Coord { x: a[0], y: a[1] }, // close
                ]),
                vec![],
            )
        })
        .collect();

    // Collect boundary segments: edges that belong to exactly one walkable triangle
    // (i.e. wall edges).
    let boundary_segs = collect_boundary_segments(&verts_f64, triangles, walkable, tri_count);
    let tree = rstar::RTree::bulk_load(boundary_segs);

    let n = width as usize * height as usize;
    let mut speed_field = vec![0.0f64; n];

    for row in 0..height {
        for col in 0..width {
            let px = origin[0] + (col as f64 + 0.5) * cell_size;
            let py = origin[1] + (row as f64 + 0.5) * cell_size;
            let pt = Point::new(px, py);

            let in_walkable = walkable_polys.iter().any(|poly| poly.contains(&pt));
            if !in_walkable {
                continue;
            }

            let dist2 = tree
                .nearest_neighbor_iter_with_distance_2(&[px, py])
                .next()
                .map(|(_, d2)| d2)
                .unwrap_or(f64::INFINITY);
            speed_field[row as usize * width as usize + col as usize] =
                (dist2.sqrt() / wall_influence_radius).clamp(0.0, 1.0);
        }
    }

    (GridParams { origin, width, height, cell_size }, speed_field)
}

/// An edge represented as a sorted (lower, upper) vertex index pair.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Edge(u32, u32);

impl Edge {
    fn new(a: u32, b: u32) -> Self {
        if a <= b { Edge(a, b) } else { Edge(b, a) }
    }
}

fn collect_boundary_segments(
    verts: &[[f64; 2]],
    triangles: &[u32],
    walkable: &[u8],
    tri_count: usize,
) -> Vec<MeshBoundarySeg> {
    use std::collections::HashMap;
    // Count how many walkable triangles each edge appears in.
    let mut edge_count: HashMap<Edge, u8> = HashMap::new();
    for t in 0..tri_count {
        if walkable[t] == 0 {
            continue;
        }
        let a = triangles[t * 3];
        let b = triangles[t * 3 + 1];
        let c = triangles[t * 3 + 2];
        *edge_count.entry(Edge::new(a, b)).or_insert(0) += 1;
        *edge_count.entry(Edge::new(b, c)).or_insert(0) += 1;
        *edge_count.entry(Edge::new(c, a)).or_insert(0) += 1;
    }
    // Boundary edges appear exactly once.
    edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(Edge(a, b), _)| MeshBoundarySeg {
            p0: verts[a as usize],
            p1: verts[b as usize],
        })
        .collect()
}

#[derive(Clone)]
struct MeshBoundarySeg {
    p0: [f64; 2],
    p1: [f64; 2],
}

impl rstar::RTreeObject for MeshBoundarySeg {
    type Envelope = rstar::AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_corners(
            [self.p0[0].min(self.p1[0]), self.p0[1].min(self.p1[1])],
            [self.p0[0].max(self.p1[0]), self.p0[1].max(self.p1[1])],
        )
    }
}

impl rstar::PointDistance for MeshBoundarySeg {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        seg_point_dist2(self.p0, self.p1, *point)
    }

    fn contains_point(&self, point: &<Self::Envelope as rstar::Envelope>::Point) -> bool {
        self.distance_2(point) == 0.0
    }
}

fn seg_point_dist2(p0: [f64; 2], p1: [f64; 2], q: [f64; 2]) -> f64 {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        let ex = q[0] - p0[0];
        let ey = q[1] - p0[1];
        return ex * ex + ey * ey;
    }
    let t = ((q[0] - p0[0]) * dx + (q[1] - p0[1]) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let cx = p0[0] + t * dx - q[0];
    let cy = p0[1] + t * dy - q[1];
    cx * cx + cy * cy
}

#[cfg(test)]
mod tests {
    use super::*;

    // A single walkable right-angle triangle with one wall along the hypotenuse.
    //
    //  (0,1)──(1,1)
    //         /
    //  (0,0)──(1,0)
    //
    // Triangle: (0,0),(1,0),(1,1) — walkable; no other triangles so all edges are boundary.
    #[test]
    fn single_triangle_walkable() {
        let vertices: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let triangles: Vec<u32> = vec![0, 1, 2];
        let walkable: Vec<u8> = vec![1];
        let (_params, speed) = build_from_mesh(&vertices, &triangles, &walkable, 0.5, 0.5);
        assert_eq!(_params.width, 2);
        assert_eq!(_params.height, 2);
        // Some cells should be walkable (non-zero).
        let any_walkable = speed.iter().any(|&s| s > 0.0);
        assert!(any_walkable, "expected at least one walkable cell");
    }

    #[test]
    fn obstacle_triangle_gives_all_zero() {
        let vertices: Vec<f64> = vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0];
        let triangles: Vec<u32> = vec![0, 1, 2];
        let walkable: Vec<u8> = vec![0]; // obstacle
        // Bounding box over zero walkable triangles → empty grid.
        let (_params, speed) = build_from_mesh(&vertices, &triangles, &walkable, 0.5, 0.5);
        assert!(speed.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn f32_vertices_accepted() {
        let vertices: Vec<f32> = vec![0.0f32, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0];
        // Two walkable triangles covering a 3×3 square.
        let triangles: Vec<u32> = vec![0, 1, 2, 0, 2, 3];
        let walkable: Vec<u8> = vec![1, 1];
        let (params, speed) = build_from_mesh(&vertices, &triangles, &walkable, 1.0, 0.5);
        assert_eq!(params.width, 3);
        assert_eq!(params.height, 3);
        let any_walkable = speed.iter().any(|&s| s > 0.0);
        assert!(any_walkable);
    }
}
