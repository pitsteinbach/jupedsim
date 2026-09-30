// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "Geometry/Geometry.hpp"
#include "RoutingEngine.hpp"
#include "SurfaceMeshShortestPathRoutingEngine.hpp"

#include <map>
#include <span>
#include <vector>

/// Hands out one id space over several routing backends and dispatches every query to the
/// backend that owns the destination.
///
/// Point destinations are routed along exact geodesics on the surface mesh. Area destinations
/// (exits) are, for now, routed by the same engine to the anchor of their nearest piece; a
/// backend that leads to the nearest part of an area plugs in here.
class HybridRoutingEngine final : public RoutingEngine
{
public:
    /// Borrows @p geometry (non-owning); the caller keeps it alive for the engine's lifetime.
    /// @param wallClearance see `SurfaceMeshShortestPathRoutingEngine`
    explicit HybridRoutingEngine(const Geometry& geometry, double wallClearance = 0.2);
    ~HybridRoutingEngine() override = default;

    using RoutingEngine::AddDestination;
    using RoutingEngine::ComputeWaypoint;
    using RoutingEngine::GetOrientation;
    using RoutingEngine::GetShortestPath;

    DestinationId AddDestination(const Point3D& point) override;
    DestinationId AddDestination(std::span<const AreaPiece> area) override;
    bool HasDestination(DestinationId id) const override;

    bool IsValidLocation(const Point3D& loc) const override;

    std::vector<Point3D> GetShortestPath(const Point3D& source, DestinationId target) override;

    Point GetOrientation(const Point3D& source, DestinationId target) override;

    Point ComputeWaypoint(const Location& from, DestinationId to) override;

private:
    /// Which backend serves an id, and the id it knows the destination by.
    struct Route {
        RoutingEngine* engine;
        DestinationId inner;
    };

    DestinationId add(RoutingEngine& engine, DestinationId inner);
    const Route& route(DestinationId id) const;

    SurfaceMeshShortestPathRoutingEngine _points;

    std::vector<Route> _routes{}; // index = DestinationId::value
    /// Registered points, so registering one again hands out the same id.
    std::map<Point3D, DestinationId> _pointIds{};
};
