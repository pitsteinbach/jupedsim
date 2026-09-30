// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Geometry/Geometry.hpp"

#include "GeometricFunctions.hpp"
#include "Geometry/BoundaryIndex.hpp"
#include "LineSegment.hpp"
#include "Polygon.hpp"
#include "SimulationError.hpp"

#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/mark_domain_in_triangulation.h>
#include <boost/iterator/function_output_iterator.hpp>
#include <boost/range/iterator_range.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <variant>
#include <vector>

Geometry::Geometry(SurfaceMesh mesh) : _mesh(std::move(mesh))
{
    // Compact vertex/face indices so vertices()/triangles()/region_id_per_face() are
    // contiguous.
    _mesh.collect_garbage();
    _regionSplit = split_into_regions(_mesh);
    build();
}

Geometry::Geometry(
    SurfaceMesh&& mesh,
    RegionSplit&& regionSplit,
    std::unique_ptr<RegionGraph2D> regionGraph2d)
    : _mesh(std::move(mesh))
    , _regionGraph2D(std::move(regionGraph2d))
    , _regionSplit(std::move(regionSplit))
{
    // safety only: hand-crafted mesh/region-split/regionGraph2d combo should not run into this
    assert(!_mesh.has_garbage());
    build();
}

void Geometry::build()
{
    _aabbTree = std::make_unique<AABBTree>(_mesh.faces().begin(), _mesh.faces().end(), _mesh);
    _region = _regionSplit.region;
    _boundaryIndex = MakePortalBoundaryIndex(_mesh, _regionSplit);
    _regionGraph = CreateRegionGraph(_mesh, _regionSplit);
}

std::optional<PolyWithHoles> Geometry::polygon() const
{
    if(_regionGraph2D && region_count() == 1) {
        return (*_regionGraph2D)[0];
    }
    return std::nullopt;
}

Geometry::FaceLocation Geometry::face_below(const Point3D& p) const
{
    // first_intersection along -z returns the hit nearest to the ray source,
    // i.e. the face directly below the query point. The ray starts a hair
    // above p: a query point sitting exactly on the surface may round minimally
    // below its face's plane, and the strictly-downward ray would miss it.
    constexpr double onSurfaceTolerance = 1e-9;
    const Ray3D ray(Point3D{p.x(), p.y(), p.z() + onSurfaceTolerance}, Direction3D(0, 0, -1));
    const auto hit = aabb_tree().first_intersection(ray);
    if(!hit) {
        return {SurfaceMesh::null_face(), K::Point_3{}};
    }
    const auto* projected = std::get_if<K::Point_3>(&hit->first);
    // Assert against vertical faces.
    assert(projected && "FATAL: vertical face hit by the face_below line");
    return {hit->second, *projected};
}

bool Geometry::is_valid_location(const Point3D& p) const
{
    return face_below(p).face != SurfaceMesh::null_face();
}

std::vector<LineSegment>
Geometry::line_segments_in_range(const Location& who, double distance) const
{
    return _boundaryIndex->Query(who, distance);
}

bool Geometry::no_geometry_between(const Location& who, Point direction) const
{
    return region_reached(who, direction).has_value();
}

bool Geometry::no_geometry_between(const Location& who, const Location& other) const
{
    const auto arrival = region_reached(who, other.xy() - who.xy());
    return arrival == other.region();
}

std::optional<std::size_t> Geometry::region_reached(const Location& who, Point direction) const
{
    const LineSegment chord{who.xy(), who.xy() + direction};
    const auto& graph = *_regionGraph;
    const auto crosses_seam = [&] {
        for(const auto e : boost::make_iterator_range(boost::out_edges(who.region(), graph))) {
            if(intersects(chord, graph[e])) {
                return true;
            }
        }
        return false;
    };
    if(!crosses_seam()) {
        // Direct line stays within this region: the region's own walls settle it.
        return graph[who.region()]->IntersectsAny(chord) ? std::nullopt :
                                                           std::optional{who.region()};
    }
    // Crosses regions: Doing the expensive "move on surface".
    const auto arrival = who.try_move_on_surface(direction);
    return arrival.has_value() ? std::optional{arrival->region()} : std::nullopt;
}

Geometry::FaceLocation Geometry::locate_in_region(std::size_t region_id, const Point2D& xy) const
{
    // All intersections along z. Search for the one with the region_id.
    const Line3D vertical(Point3D{xy.x(), xy.y(), 0}, Direction3D(0, 0, 1));
    std::vector<AABBTree::Intersection_and_primitive_id<Line3D>::Type> hits{};
    aabb_tree().all_intersections(vertical, std::back_inserter(hits));

    for(const auto& [where, face] : hits) {
        if(_region[face] != region_id) {
            continue;
        }
        const auto* point = std::get_if<Point3D>(&where);
        // Assert against vertical faces - purely defensive.
        assert(point && "FATAL: vertical face hit by the locate line");
        return {face, *point};
    }
    return {SurfaceMesh::null_face(), Point3D{}};
}

std::optional<Location> Geometry::get_location(double x, double y, double z_hint, double tol) const
{
    const auto face_location = locate_near_z(Point2D{x, y}, z_hint, tol);
    if(face_location.face == SurfaceMesh::null_face()) {
        return std::nullopt;
    }
    return Location{
        this,
        Point{x, y},
        region_of(face_location.face),
        face_location.face,
        face_location.point.z()};
}

Geometry::FaceLocation Geometry::locate_near_z(const Point2D& xy, double z, double tolerance) const
{
    const Line3D vertical(Point3D{xy.x(), xy.y(), 0}, Direction3D(0, 0, 1));
    std::vector<AABBTree::Intersection_and_primitive_id<Line3D>::Type> hits{};
    aabb_tree().all_intersections(vertical, std::back_inserter(hits));

    FaceLocation best{SurfaceMesh::null_face(), Point3D{}};
    auto bestDeviation = tolerance;
    for(const auto& [where, face] : hits) {
        const auto* point = std::get_if<Point3D>(&where);
        // Assert against vertical faces - purely defensive.
        assert(point && "FATAL: vertical face hit by the locate line");
        if(const auto deviation = std::abs(point->z() - z); deviation <= bestDeviation) {
            bestDeviation = deviation;
            best = {face, *point};
        }
    }
    return best;
}

std::vector<std::size_t> Geometry::region_id_per_face() const
{
    std::vector<std::size_t> ids{};
    ids.reserve(_mesh.number_of_faces());
    for(const auto f : _mesh.faces()) {
        ids.push_back(_region[f]);
    }
    return ids;
}

std::vector<std::array<double, 3>> Geometry::vertices() const
{
    std::vector<std::array<double, 3>> out{};
    out.reserve(_mesh.number_of_vertices());
    for(const auto v : _mesh.vertices()) {
        const auto& p = _mesh.point(v);
        out.push_back({p.x(), p.y(), p.z()});
    }
    return out;
}

std::vector<std::array<std::size_t, 3>> Geometry::triangles() const
{
    std::vector<std::array<std::size_t, 3>> out{};
    out.reserve(_mesh.number_of_faces());
    for(const auto f : _mesh.faces()) {
        std::array<std::size_t, 3> tri{};
        int i = 0;
        for(const auto v : CGAL::vertices_around_face(_mesh.halfedge(f), _mesh)) {
            tri[i++] = static_cast<std::size_t>(v);
        }
        out.push_back(tri);
    }
    return out;
}

std::vector<AreaPiece> Geometry::split_into_region_pieces(const Poly& p, double z_hint) const
{
    const auto c = Polygon(p).Centroid();
    const auto seed = get_location(c.x, c.y, z_hint);
    if(!seed) {
        throw SimulationError("Area centroid {} not on walkable surface.", c);
    }
    const auto* g = region_graph_2d();
    if(!g) {
        // No footprints to cut along: keep the polygon whole.
        return {AreaPiece{PolyWithHoles(p), seed->region()}};
    }

    std::vector<AreaPiece> pieces{};
    // define jamba function to add pieces of a region to the pieces vector
    // we use a jambda function to add the polygon and the region id to the pieces vector
    const auto add_pieces_of = [&](std::size_t r) {
        CGAL::intersection(
            (*g)[r], p, boost::make_function_output_iterator([&pieces, r](PolyWithHoles&& piece) {
                pieces.push_back(AreaPiece{std::move(piece), r});
            }));
    };

    std::vector<bool> seen(boost::num_vertices(*g), false);
    std::queue<std::size_t> todo;
    todo.push(seed->region());
    seen[seed->region()] = true;
    while(!todo.empty()) {
        const auto r = todo.front();
        todo.pop();
        add_pieces_of(r);
        for(auto e : boost::make_iterator_range(boost::out_edges(r, *g))) {
            const auto n = boost::target(e, *g);
            if(!seen[n] && CGAL::do_intersect((*g)[n], p)) {
                seen[n] = true;
                todo.push(n);
            }
        }
    }
    return pieces;
}

namespace
{
bool strictly_inside(const PolyWithHoles& area, const Point2D& p)
{
    if(area.outer_boundary().bounded_side(p) != CGAL::ON_BOUNDED_SIDE) {
        return false;
    }
    return std::none_of(area.holes_begin(), area.holes_end(), [&p](const Poly& hole) {
        return hole.bounded_side(p) != CGAL::ON_UNBOUNDED_SIDE;
    });
}
} // namespace

Location Geometry::anchor_of(const AreaPiece& piece) const
{
    const auto& outer = piece.polygon.outer_boundary();
    if(outer.is_empty()) {
        throw SimulationError("Empty area piece in region {}.", piece.region);
    }

    // Candidates, best first: the centroid, then the centre of each corner triangle - some
    // corner of a simple polygon is an ear, whose triangle lies inside - then a vertex.
    std::vector<Point2D> candidates{};
    candidates.reserve(outer.size() + 2);
    const auto c = Polygon(outer).Centroid();
    candidates.emplace_back(c.x, c.y);
    const auto n = outer.size();
    for(std::size_t i = 0; i < n; ++i) {
        candidates.push_back(CGAL::centroid(outer[(i + n - 1) % n], outer[i], outer[(i + 1) % n]));
    }
    const auto first = std::find_if(candidates.begin(), candidates.end(), [&](const auto& p) {
        return strictly_inside(piece.polygon, p);
    });
    const Point2D xy = first != candidates.end() ? *first : outer[0];

    const auto on = locate_in_region(piece.region, xy);
    if(on.face == SurfaceMesh::null_face()) {
        throw SimulationError(
            "Area piece point {} is not on region {}.", Point{xy.x(), xy.y()}, piece.region);
    }
    return Location{this, Point{xy.x(), xy.y()}, piece.region, on.face, on.point.z()};
}
