// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "CfgCgal.hpp"
#include "Point.hpp"
#include "Routing.hpp"
#include "floorfield.h"

#include <cstdint>
#include <mutex>
#include <vector>

/// Solver choice for the eikonal equation.
enum class EikonalSolver : uint8_t { FSM = 0, FMM = 1, FIM = 2 };

/// Thin C++ adapter: owns a rust::Box<jupedsim::floorfield::Floorfield> and
/// implements the Router virtual interface by delegating to it.
///
/// `Scalar` selects the internal float precision for the eikonal solver:
///   - `double` (default) uses the f64 GPU/CPU path
///   - `float`            uses the f32 GPU path
///
/// _mutex serialises all ComputeWaypoint calls: the Rust Floorfield takes
/// &mut self, so concurrent calls from the par_unseq tactical loop are UB.
template <typename Scalar = float>
class Floorfield : public Router
{
    rust::Box<jupedsim::floorfield::Floorfield> _inner;
    mutable std::mutex _mutex;

    uint32_t snapToCell(Point p) const;

public:
    explicit Floorfield(const PolyWithHoles& poly, double wallInfluenceRadius = 0.5);
    Floorfield() = delete;
    Floorfield(const Floorfield&) = delete;
    Floorfield& operator=(const Floorfield&) = delete;
    Floorfield(Floorfield&&) = delete;
    Floorfield& operator=(Floorfield&&) = delete;

    size_t AddDestination(const Poly& area) override;
    size_t AddDestination(std::span<const Poly> areas) override;

    void UpdateDensity(std::span<const Point> positions) override;
    void
    PrecomputeDestinations(std::span<const size_t> ids, std::span<const Point> points) override;

    Point ComputeWaypoint(Point currentPosition, size_t destinationId) override;
    std::vector<Point> ComputeAllWaypoints(Point currentPosition, size_t destinationId) override;
    Point ComputeWaypoint(Point currentPosition, Point destination) override;
    std::vector<Point> ComputeAllWaypoints(Point currentPosition, Point destination) override;
    bool IsRoutable(Point p) const override;
    void Update() override {}

    // Inspection / visualisation accessors (used by Python bindings).
    uint32_t GridWidth() const { return _inner->grid_width(); }
    uint32_t GridHeight() const { return _inner->grid_height(); }
    Point Origin() const { return {_inner->grid_origin_x(), _inner->grid_origin_y()}; }
    double CellSize() const { return _inner->grid_cell_size(); }
    std::vector<double> SpeedField() const;
    std::vector<double> TravelTimes() const;
    std::vector<double> DensityField() const;
    std::vector<double> DynamicSpeedField() const;
    void SetSolver(EikonalSolver s);
    void ClearPointCache();
    void SetRecomputeInterval(int steps);
};
