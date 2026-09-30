// SPDX-License-Identifier: LGPL-3.0-or-later
#include "HybridRoutingEngine.hpp"

HybridRoutingEngine::HybridRoutingEngine(const Geometry& geometry, double wallClearance)
    : RoutingEngine(wallClearance), _points(geometry, wallClearance)
{
}

DestinationId HybridRoutingEngine::add(RoutingEngine& engine, DestinationId inner)
{
    _routes.push_back({&engine, inner});
    return DestinationId{_routes.size() - 1};
}

const HybridRoutingEngine::Route& HybridRoutingEngine::route(DestinationId id) const
{
    ThrowIfUnknown(id);
    return _routes[id.value];
}

DestinationId HybridRoutingEngine::AddDestination(const Point3D& point)
{
    if(const auto it = _pointIds.find(point); it != _pointIds.end()) {
        return it->second;
    }
    const auto id = add(_points, _points.AddDestination(point));
    _pointIds.emplace(point, id);
    return id;
}

DestinationId HybridRoutingEngine::AddDestination(std::span<const AreaPiece> area)
{
    return add(_points, _points.AddDestination(area));
}

bool HybridRoutingEngine::HasDestination(DestinationId id) const
{
    return id.IsValid() && id.value < _routes.size();
}

bool HybridRoutingEngine::IsValidLocation(const Point3D& loc) const
{
    return _points.IsValidLocation(loc);
}

std::vector<Point3D>
HybridRoutingEngine::GetShortestPath(const Point3D& source, DestinationId target)
{
    const auto& r = route(target);
    return r.engine->GetShortestPath(source, r.inner);
}

Point HybridRoutingEngine::GetOrientation(const Point3D& source, DestinationId target)
{
    const auto& r = route(target);
    return r.engine->GetOrientation(source, r.inner);
}

Point HybridRoutingEngine::ComputeWaypoint(const Location& from, DestinationId to)
{
    const auto& r = route(to);
    return r.engine->ComputeWaypoint(from, r.inner);
}
