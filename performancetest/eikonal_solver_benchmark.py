"""
Eikonal solver benchmark — FSM vs FIM for point-destination floor-field
precomputation with realistic exit placement (boundary sources) across
non-square geometries and high destination counts.

Usage:
    python performancetest/eikonal_solver_benchmark.py

FMM is omitted: it is consistently 3-9× slower than FSM and does not improve
with scale. Including it would dominate runtime for large grids.

JUPEDSIM_ENABLE_PARALLEL=ON (cmake) is required for std::execution::par.
Without it all solvers run on one thread.
"""

import contextlib
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

build_lib = os.path.join(os.path.dirname(__file__), "..", "build", "lib")
sys.path.insert(0, os.path.abspath(build_lib))

import py_jupedsim as jps  # noqa: E402

# ─── Geometry definitions ──────────────────────────────────────────────────────


@dataclass
class GeometrySpec:
    label: str
    # Accessible polygon(s): list of (x, y) vertex lists
    accessible: list = field(default_factory=list)
    # Obstacle polygon(s) cut out of accessible area
    obstacles: list = field(default_factory=list)
    # Walls from which exits should be generated:
    # list of (x0,y0, x1,y1) line segments along the boundary
    exit_walls: list = field(default_factory=list)
    n_destinations: list = field(default_factory=list)
    n_reps: int = 3


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


GEOMETRIES = [
    # ── Small square — baseline ────────────────────────────────────────────
    # 100×100 m → 500×500 = 250 000 cells
    GeometrySpec(
        label="square 100×100 m",
        accessible=[rect(0, 0, 100, 100)],
        exit_walls=[
            (0, 0, 100, 0),  # south
            (100, 0, 100, 100),  # east
            (0, 100, 100, 100),  # north
            (0, 0, 0, 100),  # west
        ],
        n_destinations=[1, 4, 8, 16, 32, 64],
        n_reps=3,
    ),
    # ── Wide corridor — asymmetric aspect ratio ────────────────────────────
    # 500×20 m → 2500×100 = 250 000 cells; exits only at the short ends
    GeometrySpec(
        label="corridor 500×20 m",
        accessible=[rect(0, 0, 500, 20)],
        exit_walls=[
            (0, 0, 0, 20),  # west end
            (500, 0, 500, 20),  # east end
        ],
        n_destinations=[1, 4, 8, 16, 32, 64],
        n_reps=3,
    ),
    # ── Room with central obstacle ─────────────────────────────────────────
    # 200×200 m outer, 80×80 m obstacle in centre → ~960 000 walkable cells
    GeometrySpec(
        label="200×200 m + obstacle",
        accessible=[rect(0, 0, 200, 200)],
        obstacles=[rect(60, 60, 140, 140)],
        exit_walls=[
            (0, 0, 200, 0),
            (200, 0, 200, 200),
            (0, 200, 200, 200),
            (0, 0, 0, 200),
        ],
        n_destinations=[1, 4, 8, 16, 32, 64],
        n_reps=3,
    ),
    # ── Large non-square room ──────────────────────────────────────────────
    # 400×100 m → 2000×500 = 1 000 000 cells
    GeometrySpec(
        label="rect 400×100 m",
        accessible=[rect(0, 0, 400, 100)],
        exit_walls=[
            (0, 0, 400, 0),
            (400, 0, 400, 100),
            (0, 100, 400, 100),
            (0, 0, 0, 100),
        ],
        n_destinations=[1, 4, 8, 16, 32, 64],
        n_reps=2,
    ),
    # ── Long corridor ──────────────────────────────────────────────────────
    # 1000×20 m → 5000×100 = 500 000 cells; exits at ends + some side exits
    GeometrySpec(
        label="corridor 1000×20 m",
        accessible=[rect(0, 0, 1000, 20)],
        exit_walls=[
            (0, 0, 0, 20),
            (1000, 0, 1000, 20),
            (0, 0, 1000, 0),  # south long wall (side exits)
            (0, 20, 1000, 20),  # north long wall (side exits)
        ],
        n_destinations=[1, 4, 8, 16, 32, 64],
        n_reps=2,
    ),
    # ── Large square ──────────────────────────────────────────────────────
    # 500×500 m → 2500×2500 = 6 250 000 cells
    GeometrySpec(
        label="square 500×500 m",
        accessible=[rect(0, 0, 500, 500)],
        exit_walls=[
            (0, 0, 500, 0),
            (500, 0, 500, 500),
            (0, 500, 500, 500),
            (0, 0, 0, 500),
        ],
        n_destinations=[1, 4, 8, 16, 32],
        n_reps=2,
    ),
    # ── Very large non-square ──────────────────────────────────────────────
    # 2000×500 m → 10000×2500 = 25 000 000 cells
    GeometrySpec(
        label="rect 2000×500 m",
        accessible=[rect(0, 0, 2000, 500)],
        exit_walls=[
            (0, 0, 2000, 0),
            (2000, 0, 2000, 500),
            (0, 500, 2000, 500),
            (0, 0, 0, 500),
        ],
        n_destinations=[1, 4, 8],
        n_reps=1,
    ),
]

# FMM always skipped (proven slowest at all scales in prior benchmarks).
SOLVERS = ["FSM", "FIM"]


# ─── Helpers ──────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def suppress_cpp_stdout():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def make_floorfield(spec: GeometrySpec) -> "jps.Floorfield":
    gb = jps.GeometryBuilder()
    for poly in spec.accessible:
        gb.add_accessible_area(poly)
    for poly in spec.obstacles:
        gb.exclude_from_accessible_area(poly)
    geo = gb.build()
    with suppress_cpp_stdout():
        ff = jps.Floorfield(geo, wall_influence_radius=0.5)
    return ff


def boundary_exit_points(
    n: int, spec: GeometrySpec, rng: random.Random
) -> list:
    """
    Distribute N exit points uniformly along the exit_walls of the geometry.
    Each point is placed at distance `margin` from the wall so the cell is
    walkable (inside the wall-influence zone but not an obstacle).
    """
    margin = 0.3  # metres from wall — speed ≈ 0.6 here (walkable but slow)
    walls = spec.exit_walls
    # Compute total wall length for proportional distribution.
    lengths = []
    for x0, y0, x1, y1 in walls:
        lengths.append(math.hypot(x1 - x0, y1 - y0))
    total = sum(lengths)

    pts: set = set()
    attempts = 0
    while len(pts) < n and attempts < n * 500:
        attempts += 1
        # Pick a random wall segment (weighted by length).
        d = rng.uniform(0, total)
        cumulative = 0.0
        seg = walls[-1]
        for i, length in enumerate(lengths):
            cumulative += length
            if d <= cumulative:
                seg = walls[i]
                break

        x0, y0, x1, y1 = seg
        t = rng.uniform(0.05, 0.95)  # avoid corners
        px = x0 + t * (x1 - x0)
        py = y0 + t * (y1 - y0)

        # Nudge point inward from the wall by `margin`.
        dx, dy = x1 - x0, y1 - y0
        length_seg = math.hypot(dx, dy)
        # Normal pointing inward: rotate wall direction 90° and check sign.
        nx, ny = -dy / length_seg, dx / length_seg
        # Assume the accessible area is on the left of each wall segment as defined.
        # Simple heuristic: nudge both ways and keep the one with larger x+y centroid.
        cx = sum(v[0] for poly in spec.accessible for v in poly) / max(
            1, sum(len(poly) for poly in spec.accessible)
        )
        cy = sum(v[1] for poly in spec.accessible for v in poly) / max(
            1, sum(len(poly) for poly in spec.accessible)
        )
        # Pick inward normal direction toward centroid.
        if (px + nx * margin - cx) ** 2 + (py + ny * margin - cy) ** 2 < (
            px - nx * margin - cx
        ) ** 2 + (py - ny * margin - cy) ** 2:
            inx, iny = nx, ny
        else:
            inx, iny = -nx, -ny

        ex = round(px + inx * margin, 1)
        ey = round(py + iny * margin, 1)
        pts.add((ex, ey))

    return list(pts)[:n]


def time_precompute(
    ff: "jps.Floorfield", solver: str, points: list, n_reps: int
) -> float:
    ff.set_eikonal_solver(solver)
    with suppress_cpp_stdout():
        ff.clear_point_cache()
        ff.precompute_destinations(points)  # warm-up
        total = 0.0
        for _ in range(n_reps):
            ff.clear_point_cache()
            t0 = time.perf_counter()
            ff.precompute_destinations(points)
            total += time.perf_counter() - t0
    return total / n_reps


def fmt_time(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s "
    if seconds >= 0.001:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds * 1000:.2f} ms"


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    rng = random.Random(42)

    results: dict = {}
    n_cells_map: dict = {}

    print()
    for spec in GEOMETRIES:
        n_cells = 0  # filled after floorfield creation

        print(f"  {spec.label}", flush=True)

        ff = make_floorfield(spec)
        sf = ff.speed_field()
        n_cells = sf["width"] * sf["height"]
        n_cells_map[spec.label] = n_cells
        results[spec.label] = {}

        print(
            f"    grid {sf['width']}×{sf['height']} = {n_cells:,} cells"
            f"  reps={spec.n_reps}",
            flush=True,
        )

        for solver in SOLVERS:
            results[spec.label][solver] = {}
            for n in spec.n_destinations:
                pts = boundary_exit_points(n, spec, rng)
                actual_n = len(pts)
                t = time_precompute(ff, solver, pts, spec.n_reps)
                results[spec.label][solver][n] = (t, actual_n)
                print(
                    f"    {solver}  N={n:>2} (got {actual_n:>2})  {fmt_time(t)}",
                    flush=True,
                )

        print(flush=True)
        with suppress_cpp_stdout():
            del ff

    # ── Timing table ──────────────────────────────────────────────────────────
    print("=" * 100)
    print(
        "TIMING TABLE  —  boundary exit sources, wall-clock precomputation time"
    )
    print("=" * 100)

    for spec in GEOMETRIES:
        n_cells = n_cells_map[spec.label]
        n_dests = spec.n_destinations
        solvers = sorted(results[spec.label].keys())

        col = max(7, max(len(f"N={n}") for n in n_dests))
        hdr = f"  {'Solver':<5}  " + "  ".join(
            f"{'N=' + str(n):>{col}}" for n in n_dests
        )
        print(f"\n  {spec.label}  ({n_cells:,} cells)")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for solver in solvers:
            row = f"  {solver:<5}  "
            row += "  ".join(
                f"{fmt_time(results[spec.label][solver][n][0]):>{col}}"
                for n in n_dests
            )
            print(row)

    # ── Winner summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("WINNER SUMMARY  (speedup of FIM over FSM per N)")
    print("=" * 100)
    print()

    header = f"  {'Geometry':<26}  {'Cells':>12}  " + "  ".join(
        f"{'N=' + str(n):>8}" for n in [1, 4, 8, 16, 32, 64]
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for spec in GEOMETRIES:
        n_cells = n_cells_map[spec.label]
        row = f"  {spec.label:<26}  {n_cells:>12,}  "
        parts = []
        for n in [1, 4, 8, 16, 32, 64]:
            if n not in spec.n_destinations:
                parts.append(f"{'—':>8}")
                continue
            t_fsm = results[spec.label]["FSM"][n][0]
            t_fim = results[spec.label]["FIM"][n][0]
            speedup = t_fsm / t_fim
            parts.append(f"{speedup:>7.2f}×")
        row += "  ".join(parts)
        print(row)

    print()


if __name__ == "__main__":
    main()
