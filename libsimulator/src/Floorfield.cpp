// SPDX-License-Identifier: LGPL-3.0-or-later

#include "Floorfield.hpp"

#include "CfgCgal.hpp"
#include "SimulationError.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace
{

/// Serialise a CGAL Poly ring into a flat x,y double vector.
std::vector<double> ringToFlat(const Poly& ring)
{
    std::vector<double> out;
    out.reserve(ring.size() * 2);
    for(auto v = ring.vertices_begin(); v != ring.vertices_end(); ++v) {
        out.push_back(CGAL::to_double(v->x()));
        out.push_back(CGAL::to_double(v->y()));
    }
    return out;
}

/// rust::Box has no default constructor, so we must construct _inner in the
/// member-initializer list.  `Scalar` selects which Rust constructor to call.
template <typename Scalar>
rust::Box<jupedsim::floorfield::Floorfield>
buildInner(const PolyWithHoles& poly, double cellSize, double wallInfluenceRadius)
{
    const auto& outer = poly.outer_boundary();
    auto outerFlat = ringToFlat(outer);

    std::vector<double> holesFlat;
    std::vector<uint32_t> holeLengths;
    for(auto h = poly.holes_begin(); h != poly.holes_end(); ++h) {
        auto hflat = ringToFlat(*h);
        holeLengths.push_back(static_cast<uint32_t>(hflat.size()));
        holesFlat.insert(holesFlat.end(), hflat.begin(), hflat.end());
    }

    const rust::Slice<const double> outerSlice{outerFlat.data(), outerFlat.size()};
    const rust::Slice<const double> holesSlice{holesFlat.data(), holesFlat.size()};
    const rust::Slice<const uint32_t> lengthsSlice{holeLengths.data(), holeLengths.size()};

    if constexpr(std::is_same_v<Scalar, float>) {
        return jupedsim::floorfield::new_floorfield_f32_from_polygon(
            outerSlice, holesSlice, lengthsSlice, cellSize, wallInfluenceRadius);
    } else {
        return jupedsim::floorfield::new_floorfield_f64_from_polygon(
            outerSlice, holesSlice, lengthsSlice, cellSize, wallInfluenceRadius);
    }
}

} // namespace

// ── Constructor ───────────────────────────────────────────────────────────────

template <typename Scalar>
Floorfield<Scalar>::Floorfield(const PolyWithHoles& poly, double wallInfluenceRadius)
    : _inner(buildInner<Scalar>(poly, 0.2, wallInfluenceRadius))
{
}

// ── snapToCell ────────────────────────────────────────────────────────────────

template <typename Scalar>
uint32_t Floorfield<Scalar>::snapToCell(Point p) const
{
    const auto cs = _inner->grid_cell_size();
    const auto ox = _inner->grid_origin_x();
    const auto oy = _inner->grid_origin_y();
    const auto gw = static_cast<int>(_inner->grid_width());
    const auto gh = static_cast<int>(_inner->grid_height());
    const int col = std::clamp(static_cast<int>((p.x - ox) / cs), 0, gw - 1);
    const int row = std::clamp(static_cast<int>((p.y - oy) / cs), 0, gh - 1);
    return static_cast<uint32_t>(row) * static_cast<uint32_t>(gw) + static_cast<uint32_t>(col);
}

// ── AddDestination ────────────────────────────────────────────────────────────

template <typename Scalar>
size_t Floorfield<Scalar>::AddDestination(const Poly& area)
{
    return AddDestination(std::span<const Poly>{&area, 1});
}

template <typename Scalar>
size_t Floorfield<Scalar>::AddDestination(std::span<const Poly> areas)
{
    // Rasterise all areas using CGAL point-in-polygon, then pass the merged
    // cell indices to Rust (avoids re-implementing CGAL geometry in Rust for
    // the C++ path — Rust's add_destination_cells just stores the indices).
    const auto gridWidth = _inner->grid_width();
    const auto gridHeight = _inner->grid_height();
    const auto cellSize = _inner->grid_cell_size();
    const auto originX = _inner->grid_origin_x();
    const auto originY = _inner->grid_origin_y();

    std::vector<uint32_t> cells;
    for(const auto& area : areas) {
        for(uint32_t row = 0; row < gridHeight; ++row) {
            for(uint32_t col = 0; col < gridWidth; ++col) {
                const K::Point_2 p{
                    originX + (col + 0.5) * cellSize, originY + (row + 0.5) * cellSize};
                if(area.bounded_side(p) == CGAL::ON_BOUNDED_SIDE) {
                    cells.push_back(row * gridWidth + col);
                }
            }
        }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());

    if(cells.empty()) {
        // Every polygon smaller than one cell — snap each centroid and merge.
        for(const auto& area : areas) {
            double sx = 0, sy = 0;
            for(auto v = area.vertices_begin(); v != area.vertices_end(); ++v) {
                sx += CGAL::to_double(v->x());
                sy += CGAL::to_double(v->y());
            }
            const double n = static_cast<double>(area.size());
            cells.push_back(snapToCell({sx / n, sy / n}));
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    }

    return _inner->add_destination_cells(rust::Slice<const uint32_t>{cells.data(), cells.size()});
}

// ── UpdateDensity ─────────────────────────────────────────────────────────────

template <typename Scalar>
void Floorfield<Scalar>::UpdateDensity(std::span<const Point> positions)
{
    std::vector<double> xy;
    xy.reserve(positions.size() * 2);
    for(const auto& p : positions) {
        xy.push_back(p.x);
        xy.push_back(p.y);
    }
    _inner->update_density(rust::Slice<const double>{xy.data(), xy.size()});
}

// ── PrecomputeDestinations ────────────────────────────────────────────────────

template <typename Scalar>
void Floorfield<Scalar>::PrecomputeDestinations(
    std::span<const size_t> /*ids*/,
    std::span<const Point> points)
{
    std::vector<double> xy;
    xy.reserve(points.size() * 2);
    for(const auto& p : points) {
        xy.push_back(p.x);
        xy.push_back(p.y);
    }
    _inner->precompute_destinations(rust::Slice<const double>{xy.data(), xy.size()});
}

// ── IsRoutable ────────────────────────────────────────────────────────────────

template <typename Scalar>
bool Floorfield<Scalar>::IsRoutable(Point p) const
{
    return _inner->is_routable(p.x, p.y);
}

// ── ComputeWaypoint (polygon destination) ─────────────────────────────────────

template <typename Scalar>
Point Floorfield<Scalar>::ComputeWaypoint(Point currentPosition, size_t destinationId)
{
    std::lock_guard<std::mutex> lock(_mutex);
    const auto wp =
        _inner->compute_waypoint_dest(currentPosition.x, currentPosition.y, destinationId);
    return {wp.x, wp.y};
}

template <typename Scalar>
std::vector<Point>
Floorfield<Scalar>::ComputeAllWaypoints(Point currentPosition, size_t destinationId)
{
    const auto wps =
        _inner->compute_all_waypoints_dest(currentPosition.x, currentPosition.y, destinationId);
    std::vector<Point> out;
    out.reserve(wps.size());
    for(const auto& wp : wps) {
        out.push_back({wp.x, wp.y});
    }
    return out;
}

// ── ComputeWaypoint (point destination) ──────────────────────────────────────

template <typename Scalar>
Point Floorfield<Scalar>::ComputeWaypoint(Point currentPosition, Point destination)
{
    std::lock_guard<std::mutex> lock(_mutex);
    const auto wp = _inner->compute_waypoint_point(
        currentPosition.x, currentPosition.y, destination.x, destination.y);
    return {wp.x, wp.y};
}

template <typename Scalar>
std::vector<Point> Floorfield<Scalar>::ComputeAllWaypoints(Point currentPosition, Point destination)
{
    const auto wps = _inner->compute_all_waypoints_point(
        currentPosition.x, currentPosition.y, destination.x, destination.y);
    std::vector<Point> out;
    out.reserve(wps.size());
    for(const auto& wp : wps) {
        out.push_back({wp.x, wp.y});
    }
    return out;
}

// ── Inspection / visualisation accessors ─────────────────────────────────────

namespace
{
template <typename RustVec>
std::vector<double> rustVecToStd(RustVec rv)
{
    return std::vector<double>(rv.begin(), rv.end());
}
} // namespace

template <typename Scalar>
std::vector<double> Floorfield<Scalar>::SpeedField() const
{
    return rustVecToStd(_inner->get_speed_field());
}

template <typename Scalar>
std::vector<double> Floorfield<Scalar>::TravelTimes() const
{
    return rustVecToStd(_inner->get_travel_times());
}

template <typename Scalar>
std::vector<double> Floorfield<Scalar>::DensityField() const
{
    return rustVecToStd(_inner->get_density_field());
}

template <typename Scalar>
std::vector<double> Floorfield<Scalar>::DynamicSpeedField() const
{
    return rustVecToStd(_inner->get_dynamic_speed_field());
}

template <typename Scalar>
void Floorfield<Scalar>::SetSolver(EikonalSolver s)
{
    _inner->set_solver(static_cast<uint8_t>(s));
}

template <typename Scalar>
void Floorfield<Scalar>::ClearPointCache()
{
    _inner->clear_point_cache();
}

template <typename Scalar>
void Floorfield<Scalar>::SetRecomputeInterval(int steps)
{
    _inner->set_recompute_interval(steps);
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template class Floorfield<double>;
template class Floorfield<float>;
