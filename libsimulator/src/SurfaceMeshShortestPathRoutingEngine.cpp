// SPDX-License-Identifier: LGPL-3.0-or-later
#include "SurfaceMeshShortestPathRoutingEngine.hpp"

#include "SimulationError.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// SurfaceMeshShortestPathRoutingEngine
////////////////////////////////////////////////////////////////////////////////
SurfaceMeshShortestPathRoutingEngine::SurfaceMeshShortestPathRoutingEngine(
    const Geometry& geometry,
    double wallClearance)
    : RoutingEngine(wallClearance), _geometry(geometry)
{
}

DestinationId SurfaceMeshShortestPathRoutingEngine::add(std::vector<Point3D> sources)
{
    _destinations.push_back({std::move(sources)});
    return DestinationId{_destinations.size() - 1};
}

DestinationId SurfaceMeshShortestPathRoutingEngine::AddDestination(const Point3D& point)
{
    if(const auto it = _pointIds.find(point); it != _pointIds.end()) {
        return it->second;
    }
    // Keep the exact (x, y) the caller asked for, at the height of the surface there.
    const auto below = on_surface(point, "target");
    const auto id = add({Point3D{point.x(), point.y(), below.point.z()}});
    _pointIds.emplace(point, id);
    return id;
}

DestinationId SurfaceMeshShortestPathRoutingEngine::AddDestination(std::span<const AreaPiece> area)
{
    if(area.empty()) {
        throw SimulationError("An area destination needs at least one piece.");
    }
    std::vector<Point3D> anchors{};
    anchors.reserve(area.size());
    for(const auto& piece : area) {
        anchors.push_back(_geometry.anchor_of(piece).position_3d());
    }
    return add(std::move(anchors));
}

bool SurfaceMeshShortestPathRoutingEngine::HasDestination(DestinationId id) const
{
    return id.IsValid() && id.value < _destinations.size();
}

bool SurfaceMeshShortestPathRoutingEngine::IsValidLocation(const Point3D& loc) const
{
    return _geometry.face_below(loc).face != SurfaceMesh::null_face();
}

Geometry::FaceLocation
SurfaceMeshShortestPathRoutingEngine::on_surface(const Point3D& p, const char* what) const
{
    const auto below = _geometry.face_below(p);
    if(below.face == SurfaceMesh::null_face()) {
        throw SimulationError(
            "GetShortestPath(): {} does not project onto the walkable surface.", what);
    }
    return below;
}

SurfaceMeshShortestPathRoutingEngine::ShortestPath&
SurfaceMeshShortestPathRoutingEngine::tree_for(DestinationId target)
{
    ThrowIfUnknown(target);
    auto& dest = _destinations[target.value];
    if(!dest.tree) {
        auto tree = std::make_unique<ShortestPath>(_geometry.mesh());
        for(const auto& source : dest.sources) {
            tree->add_source_point(tree->locate(source, _geometry.aabb_tree()));
        }
        tree->build_sequence_tree();
        dest.tree = std::move(tree);
    }
    return *dest.tree;
}

SurfaceMeshShortestPathRoutingEngine::Way
SurfaceMeshShortestPathRoutingEngine::trace_way(const Point3D& source, DestinationId target)
{
    const auto from_below = on_surface(source, "source");
    auto& tree = tree_for(target);
    const auto from_loc = tree.locate(from_below.point, _geometry.aabb_tree());

    // The very points `shortest_path_points_to_source_points` gives -- that call is this walk
    // with the kind of simplex thrown away. Keeping it is what tells a corner from a crossing.
    struct Collector {
        ShortestPath& tree;
        const SurfaceMesh& mesh;
        double clearance;
        Way& way;

        Point xy(SurfaceMesh::Vertex_index v) const
        {
            const auto& p = mesh.point(v);
            return Point{p.x(), p.y()};
        }

        bool on_wall(SurfaceMesh::Vertex_index v) const
        {
            return CGAL::is_border(v, mesh).has_value();
        }

        /// Where the open lies at a corner: opposite the two walls that meet there.
        ///
        /// Both walls end at the corner, so a point on their bisector is that far from either.
        /// Which of the two directions is the open one needs no test: a geodesic only turns
        /// around a corner the walkable area bends away from, so the free side is always the
        /// wider one. Judged in plan, like everything a model is told.
        Point out_of_the_corner(SurfaceMesh::Vertex_index v) const
        {
            const auto border = CGAL::is_border(v, mesh);
            if(!border) {
                // An interior vertex: the way bends over a fold, there is no wall to avoid.
                return {0.0, 0.0};
            }
            const Point here = xy(v);
            const Point along = (xy(mesh.source(*border)) - here).Normalized();
            const Point onwards = (xy(mesh.target(mesh.next(*border))) - here).Normalized();
            // A straight wall running through has no corner to step around.
            return (along + onwards).Normalized() * -1.0;
        }

        /// The crossing of @p e, held off whichever of its ends is a wall. CGAL weights the
        /// edge's target: the point is `t * target + (1 - t) * source`.
        K::FT held_off_the_walls(SurfaceMesh::Halfedge_index e, K::FT t) const
        {
            const double length = (xy(mesh.target(e)) - xy(mesh.source(e))).Norm();
            const double margin = std::min(clearance / length, 0.5);
            const double low = on_wall(mesh.source(e)) ? margin : 0.0;
            const double high = on_wall(mesh.target(e)) ? 1.0 - margin : 1.0;
            return std::clamp(CGAL::to_double(t), low, high);
        }

        void operator()(SurfaceMesh::Halfedge_index e, K::FT t)
        {
            way.push_back({tree.point(e, held_off_the_walls(e, t)), Point{0.0, 0.0}});
        }
        void operator()(SurfaceMesh::Vertex_index v)
        {
            way.push_back({tree.point(v), out_of_the_corner(v)});
        }
        void operator()(SurfaceMesh::Face_index f, const ShortestPath::Barycentric_coordinates& bc)
        {
            way.push_back({tree.point(f, bc), Point{0.0, 0.0}});
        }
    };

    Way way{};
    Collector collector{tree, _geometry.mesh(), WallClearance(), way};
    tree.shortest_path_sequence_to_source_points(from_loc.first, from_loc.second, collector);
    return way;
}

Point3D SurfaceMeshShortestPathRoutingEngine::held_off_the_wall(
    const Point3D& corner,
    Point into_the_open) const
{
    const Point moved = Point{corner.x(), corner.y()} + into_the_open * WallClearance();
    // Back onto the surface: beside a corner the floor may climb, and the route has to stay on
    // the storey the corner belongs to. Half a metre is far more than the clearance can climb
    // and far less than one storey.
    constexpr double sameStorey = 0.5;
    const auto located = _geometry.get_location(moved.x, moved.y, corner.z(), sameStorey);
    return located ? located->position_3d() : Point3D{moved.x, moved.y, corner.z()};
}

std::vector<Point3D>
SurfaceMeshShortestPathRoutingEngine::GetShortestPath(const Point3D& source, DestinationId target)
{
    const auto way = trace_way(source, target);

    std::vector<Point3D> path{};
    path.reserve(way.size());
    for(const auto& step : way) {
        path.push_back(
            step.intoTheOpen.isZeroLength() ? step.point :
                                              held_off_the_wall(step.point, step.intoTheOpen));
    }
    return path;
}

Point SurfaceMeshShortestPathRoutingEngine::next_waypoint(
    const Point3D& source,
    DestinationId target)
{
    const Point here{source.x(), source.y()};
    // CGAL sets a point wherever the way crosses a triangle edge.
    for(const auto& p : GetShortestPath(source, target)) {
        const Point xy{p.x(), p.y()};
        if(!(xy - here).isZeroLength()) {
            // Return first point "far enough" from source
            return xy;
        }
    }
    return here;
}

Point SurfaceMeshShortestPathRoutingEngine::ComputeWaypoint(const Location& from, DestinationId to)
{
    const Point next = next_waypoint(from.position_3d(), to);
    if(next != from.xy()) {
        return next;
    }
    // Nothing left to walk: already there. Head for the nearest source itself.
    const auto& sources = _destinations[to.value].sources;
    const auto nearest = std::min_element(
        sources.begin(), sources.end(), [&from](const Point3D& a, const Point3D& b) {
            return CGAL::squared_distance(a, from.position_3d()) <
                   CGAL::squared_distance(b, from.position_3d());
        });
    return Point{nearest->x(), nearest->y()};
}

Point SurfaceMeshShortestPathRoutingEngine::GetOrientation(
    const Point3D& source,
    DestinationId target)
{
    const Point here{source.x(), source.y()};
    // Zero when the way heads for where it already is: nowhere left to go.
    return (next_waypoint(source, target) - here).Normalized();
}
