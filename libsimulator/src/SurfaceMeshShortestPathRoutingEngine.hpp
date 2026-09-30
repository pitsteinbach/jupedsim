// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "Geometry/Geometry.hpp"
#include "RoutingEngine.hpp"

#include <CGAL/Surface_mesh_shortest_path.h>

#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

/// Exact geodesics on the walkable surface mesh (CGAL `Surface_mesh_shortest_path`).
///
/// An area destination is routed to the anchor point of its nearest piece
/// (`Geometry::anchor_of`), not to the nearest point of the area.
class SurfaceMeshShortestPathRoutingEngine : public RoutingEngine
{
public:
    /// Borrows @p geometry (non-owning); the caller keeps it alive for the
    /// engine's lifetime. Ownership lives with the world (later: Simulation),
    /// matching the 2D pipeline where engines never own the geometry.
    explicit SurfaceMeshShortestPathRoutingEngine(
        const Geometry& geometry,
        double wallClearance = 0.2);
    ~SurfaceMeshShortestPathRoutingEngine() override = default;

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
    using Traits = CGAL::Surface_mesh_shortest_path_traits<K, SurfaceMesh>;
    using ShortestPath = CGAL::Surface_mesh_shortest_path<Traits>;

    struct Step {
        Point3D point;
        /// Unit vector off a wall corner, zero where there is no corner.
        Point intoTheOpen;
    };
    using Way = std::vector<Step>;

    /// Where @p p sits on the surface. Throws naming @p what if it sits nowhere.
    Geometry::FaceLocation on_surface(const Point3D& p, const char* what) const;

    /// The sequence tree for @p target: built on first use, then kept.
    ShortestPath& tree_for(DestinationId target);

    Way trace_way(const Point3D& source, DestinationId target);

    /// @p corner moved into the open by the wall clearance, put back onto the surface.
    Point3D held_off_the_wall(const Point3D& corner, Point into_the_open) const;

    /// Next point of the path from @p source to @p target
    /// Returns @p source itself when @p target is already reached.
    Point next_waypoint(const Point3D& source, DestinationId target);

    DestinationId add(std::vector<Point3D> sources);

    const Geometry& _geometry;

    struct Destination {
        /// On-surface points the tree grows from; a route ends at the nearest one.
        std::vector<Point3D> sources;
        std::unique_ptr<ShortestPath> tree{};
    };
    std::vector<Destination> _destinations{}; // index = DestinationId::value
    /// Registered points, so registering one again hands out the same id.
    std::map<Point3D, DestinationId> _pointIds{};
};
