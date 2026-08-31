// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CollisionGeometry.hpp"
#include "Floorfield.hpp"
#include "RoutingEngine.hpp"
#include "conversion.hpp"

#include <glm/ext/vector_float2.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace py = pybind11;

void init_routing(py::module_& m)
{
    using Floorfield = ::Floorfield<>; // concrete instantiation with default Scalar
    py::class_<RoutingEngine>(m, "RoutingEngine")
        .def(py::init([](const CollisionGeometry& geo) {
            return std::make_unique<RoutingEngine>(geo.Polygon());
        }))
        .def(
            "compute_waypoints",
            [](RoutingEngine& engine,
               std::tuple<double, double> from,
               std::tuple<double, double> to) {
                return intoTuples(engine.ComputeAllWaypoints(intoPoint(from), intoPoint(to)));
            })
        .def(
            "is_routable",
            [](RoutingEngine& engine, std::tuple<double, double> point) {
                return engine.IsRoutable(intoPoint(point));
            })
        .def("mesh", [](const RoutingEngine& routingEngine) {
            const auto mesh = routingEngine.MeshData();
            const auto polygonCount = mesh->CountPolygons();
            using Ind = decltype(mesh->Polygons(0).vertices);
            std::vector<Ind> polys;
            polys.reserve(polygonCount);
            for(size_t index = 0; index < polygonCount; ++index) {
                const auto& poly = mesh->Polygons(index);
                polys.emplace_back(poly.vertices);
            }
            return std::make_tuple(intoTuples(mesh->FVertices()), polys);
        });

    py::class_<Floorfield>(m, "Floorfield")
        .def(
            py::init([](const CollisionGeometry& geo, double wallInfluenceRadius) {
                return std::make_unique<Floorfield>(geo.Polygon(), wallInfluenceRadius);
            }),
            py::arg("geometry"),
            py::arg("wall_influence_radius") = 0.5)
        .def(
            "compute_waypoints",
            [](Floorfield& ff, std::tuple<double, double> from, std::tuple<double, double> to) {
                return intoTuples(ff.ComputeAllWaypoints(intoPoint(from), intoPoint(to)));
            })
        .def(
            "is_routable",
            [](Floorfield& ff, std::tuple<double, double> point) {
                return ff.IsRoutable(intoPoint(point));
            })
        .def(
            "speed_field",
            [](const Floorfield& ff) {
                // Returns a dict ready for numpy: reshape data as
                // np.array(d["data"]).reshape(d["height"], d["width"])
                // Extent for imshow: [origin_x, origin_x + width*cell_size,
                //                      origin_y, origin_y + height*cell_size]
                py::dict d;
                d["width"] = ff.GridWidth();
                d["height"] = ff.GridHeight();
                d["origin"] = std::make_tuple(ff.Origin().x, ff.Origin().y);
                d["cell_size"] = ff.CellSize();
                d["data"] = ff.SpeedField();
                return d;
            })
        .def(
            "travel_times",
            [](const Floorfield& ff) {
                // Same layout as speed_field; call compute_waypoints first to
                // populate the field for a specific destination.
                py::dict d;
                d["width"] = ff.GridWidth();
                d["height"] = ff.GridHeight();
                d["origin"] = std::make_tuple(ff.Origin().x, ff.Origin().y);
                d["cell_size"] = ff.CellSize();
                d["data"] = ff.TravelTimes();
                return d;
            })
        .def(
            "set_eikonal_solver",
            [](Floorfield& ff, const std::string& name) {
                if(name == "FSM")
                    ff.SetSolver(EikonalSolver::FSM);
                else if(name == "FMM")
                    ff.SetSolver(EikonalSolver::FMM);
                else if(name == "FIM")
                    ff.SetSolver(EikonalSolver::FIM);
                else
                    throw std::invalid_argument("Unknown solver: " + name + " (use FSM, FMM, FIM)");
            },
            py::arg("solver"),
            "Select the eikonal solver: 'FSM', 'FMM', or 'FIM'. Disables auto-benchmark.")
        .def(
            "clear_point_cache",
            [](Floorfield& ff) { ff.ClearPointCache(); },
            "Clear the point-destination floor-field cache and reset the step counter.")
        .def(
            "update_density",
            [](Floorfield& ff, std::vector<std::tuple<double, double>> positions) {
                std::vector<Point> pts;
                pts.reserve(positions.size());
                for(const auto& p : positions)
                    pts.push_back(intoPoint(p));
                ff.UpdateDensity(std::span<const Point>(pts));
            },
            py::arg("positions"),
            "Update the density field from agent positions. Rebuilds dynamic speed field every "
            "set_recompute_interval() calls.")
        .def(
            "density_field",
            [](const Floorfield& ff) {
                py::dict d;
                d["width"] = ff.GridWidth();
                d["height"] = ff.GridHeight();
                d["origin"] = std::make_tuple(ff.Origin().x, ff.Origin().y);
                d["cell_size"] = ff.CellSize();
                d["data"] = ff.DensityField();
                return d;
            },
            "Returns raw agent density (agents/m²) per cell. Same dict layout as speed_field().")
        .def(
            "dynamic_speed_field",
            [](const Floorfield& ff) {
                py::dict d;
                d["width"] = ff.GridWidth();
                d["height"] = ff.GridHeight();
                d["origin"] = std::make_tuple(ff.Origin().x, ff.Origin().y);
                d["cell_size"] = ff.CellSize();
                d["data"] = ff.DynamicSpeedField();
                return d;
            },
            "Returns density-modulated speed field (0–1) per cell. Call update_density() first.")
        .def(
            "set_recompute_interval",
            [](Floorfield& ff, int steps) { ff.SetRecomputeInterval(steps); },
            py::arg("steps"),
            "Rebuild density/dynamic-speed fields every *steps* calls to update_density().")
        .def(
            "precompute_destinations",
            [](Floorfield& ff, std::vector<std::tuple<double, double>> points) {
                std::vector<Point> pts;
                pts.reserve(points.size());
                for(const auto& p : points)
                    pts.push_back(intoPoint(p));
                std::vector<size_t> ids;
                ff.PrecomputeDestinations(
                    std::span<const size_t>(ids), std::span<const Point>(pts));
            },
            py::arg("points"),
            "Precompute floor fields for the given list of (x, y) point destinations.");
}
