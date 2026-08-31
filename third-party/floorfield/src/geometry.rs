use geo::{BoundingRect, Contains, Coord, LineString, Point, Polygon};
use rstar::{PointDistance, RTreeObject, AABB};

pub struct GridParams {
    pub origin: [f64; 2],
    pub width: u32,
    pub height: u32,
    pub cell_size: f64,
}

/// Build a speed field grid from a polygon with holes.
///
/// `outer`: x,y interleaved vertices of the outer boundary.
/// `holes`: all hole vertices concatenated x,y interleaved.
/// `hole_lengths`: number of f64 values (2 × vertex count) per hole ring.
/// Returns grid parameters and a row-major speed field where 0.0 = obstacle
/// and (0, 1] = walkable, linearly reduced near walls up to `wall_influence_radius`.
pub fn build_from_polygon(
    outer: &[f64],
    holes: &[f64],
    hole_lengths: &[u32],
    cell_size: f64,
    wall_influence_radius: f64,
) -> (GridParams, Vec<f64>) {
    let exterior: Vec<Coord<f64>> = outer
        .chunks_exact(2)
        .map(|c| Coord { x: c[0], y: c[1] })
        .collect();

    let mut offset = 0usize;
    let hole_rings: Vec<LineString<f64>> = hole_lengths
        .iter()
        .map(|&len| {
            let end = offset + len as usize;
            let coords: Vec<Coord<f64>> = holes[offset..end]
                .chunks_exact(2)
                .map(|c| Coord { x: c[0], y: c[1] })
                .collect();
            offset = end;
            LineString::new(coords)
        })
        .collect();

    let polygon = Polygon::new(LineString::new(exterior), hole_rings);

    let bbox = polygon
        .bounding_rect()
        .expect("polygon must have at least one vertex");
    let origin = [bbox.min().x, bbox.min().y];
    let width = ((bbox.max().x - bbox.min().x) / cell_size).ceil() as u32;
    let height = ((bbox.max().y - bbox.min().y) / cell_size).ceil() as u32;

    let segments = collect_boundary_segments(outer, holes, hole_lengths);
    let tree = rstar::RTree::bulk_load(segments);

    let n = width as usize * height as usize;
    let mut speed_field = vec![0.0f64; n];

    for row in 0..height {
        for col in 0..width {
            let px = origin[0] + (col as f64 + 0.5) * cell_size;
            let py = origin[1] + (row as f64 + 0.5) * cell_size;
            if polygon.contains(&Point::new(px, py)) {
                let dist2 = tree
                    .nearest_neighbor_iter_with_distance_2(&[px, py])
                    .next()
                    .map(|(_, d2)| d2)
                    .unwrap_or(f64::INFINITY);
                speed_field[row as usize * width as usize + col as usize] =
                    (dist2.sqrt() / wall_influence_radius).clamp(0.0, 1.0);
            }
        }
    }

    (
        GridParams {
            origin,
            width,
            height,
            cell_size,
        },
        speed_field,
    )
}

fn collect_boundary_segments(
    outer: &[f64],
    holes: &[f64],
    hole_lengths: &[u32],
) -> Vec<BoundarySeg> {
    let mut segs = Vec::new();
    add_ring_segments(&mut segs, outer);
    let mut offset = 0usize;
    for &len in hole_lengths {
        let end = offset + len as usize;
        add_ring_segments(&mut segs, &holes[offset..end]);
        offset = end;
    }
    segs
}

fn add_ring_segments(segs: &mut Vec<BoundarySeg>, ring: &[f64]) {
    let pts: Vec<[f64; 2]> = ring.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    if pts.len() < 2 {
        return;
    }
    for w in pts.windows(2) {
        segs.push(BoundarySeg { p0: w[0], p1: w[1] });
    }
    segs.push(BoundarySeg {
        p0: *pts.last().unwrap(),
        p1: pts[0],
    });
}

#[derive(Clone)]
struct BoundarySeg {
    p0: [f64; 2],
    p1: [f64; 2],
}

impl RTreeObject for BoundarySeg {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.p0[0].min(self.p1[0]), self.p0[1].min(self.p1[1])],
            [self.p0[0].max(self.p1[0]), self.p0[1].max(self.p1[1])],
        )
    }
}

impl PointDistance for BoundarySeg {
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
