// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "CfgCgal.hpp"
#include "Geometry/Location.hpp"
#include "Point.hpp"
#include "SimulationError.hpp"

#include <compare>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

/// Handle of a destination registered with a routing engine. Only meaningful to the engine
/// that handed it out. Default-constructed it names no destination.
struct DestinationId {
    std::size_t value{std::numeric_limits<std::size_t>::max()};

    bool IsValid() const { return value != std::numeric_limits<std::size_t>::max(); }
    auto operator<=>(const DestinationId&) const = default;
};

/// Pure interface for 3D routing engines.
///
/// Destinations are registered once - a point, or an area made of pieces that each lie in one
/// region - and queried by their id from then on.
class RoutingEngine
{
public:
    /// @param wallClearance how far a route is held off the wall corners it turns on.
    explicit RoutingEngine(double wallClearance = 0.2) : _wallClearance(wallClearance)
    {
        if(wallClearance < 0.0) {
            throw SimulationError("Wall clearance cannot be negative, got {}.", wallClearance);
        }
    }
    virtual ~RoutingEngine() = default;

    // Non-copyable and non-movable
    RoutingEngine(const RoutingEngine&) = delete;
    RoutingEngine& operator=(const RoutingEngine&) = delete;
    RoutingEngine(RoutingEngine&&) = delete;
    RoutingEngine& operator=(RoutingEngine&&) = delete;

    // -- Registration ---------------------------------------------------------

    /// Register a point destination. Registering the same point again returns the same id.
    /// Throws if @p point does not project onto the walkable surface.
    virtual DestinationId AddDestination(const Point3D& point) = 0;

    DestinationId AddDestination(const Location& point)
    {
        return AddDestination(point.position_3d());
    }

    /// Register one destination made of all @p area pieces: routes lead to the nearest one.
    virtual DestinationId AddDestination(std::span<const AreaPiece> area) = 0;

    /// Whether @p id was handed out by this engine.
    virtual bool HasDestination(DestinationId id) const = 0;

    // -- Queries --------------------------------------------------------------

    /// Checks whether the provided 3D-point projects onto the walkable surface.
    virtual bool IsValidLocation(const Point3D& loc) const = 0;

    /// Compute the shortest path from @p source to @p target, held off wall corners by the
    /// engine's wall clearance.
    /// @return the path, including source as first and target as last element
    virtual std::vector<Point3D> GetShortestPath(const Point3D& source, DestinationId target) = 0;

    /// Get orientation to next point of the shortest path from @p source to
    /// @p target, projected to x/y. Zero once there is nowhere left to go.
    virtual Point GetOrientation(const Point3D& source, DestinationId target) = 0;

    /// The very next point to head for from @p from towards @p to, projected to x/y.
    ///
    /// Interim: The idea is to move to `GetOrientation`.
    virtual Point ComputeWaypoint(const Location& from, DestinationId to) = 0;

    // -- Point-to-point convenience: register (deduplicated), then query -------
    // Derived classes re-expose these with `using RoutingEngine::...`.

    std::vector<Point3D> GetShortestPath(const Point3D& source, const Point3D& target)
    {
        return GetShortestPath(source, AddDestination(target));
    }

    Point GetOrientation(const Point3D& source, const Point3D& target)
    {
        return GetOrientation(source, AddDestination(target));
    }

    Point ComputeWaypoint(const Location& from, const Location& to)
    {
        return ComputeWaypoint(from, AddDestination(to));
    }

    double WallClearance() const { return _wallClearance; }

protected:
    /// Throws unless @p id was handed out by this engine.
    void ThrowIfUnknown(DestinationId id) const
    {
        if(!HasDestination(id)) {
            throw SimulationError("Unknown routing destination {}.", id.value);
        }
    }

private:
    double _wallClearance;
};
