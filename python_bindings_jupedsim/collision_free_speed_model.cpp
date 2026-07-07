// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CollisionFreeSpeedModel.hpp"
#include "OperationalModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

void init_collision_free_speed_model(py::module_& m)
{
    py::class_<CollisionFreeSpeedModel, OperationalModel, py::smart_holder>(
        m, "CollisionFreeSpeedModel")
        .def(py::init<>());
    const CollisionFreeSpeedModel::Agent d{};
    py::class_<CollisionFreeSpeedModel::Agent>(m, "CollisionFreeSpeedModelState")
        .def(
            py::init([](Point position,
                        Point orientation,
                        double timeGap,
                        double desiredSpeed,
                        double radius,
                        double strengthNeighborRepulsion,
                        double rangeNeighborRepulsion,
                        double strengthGeometryRepulsion,
                        double rangeGeometryRepulsion) {
                return CollisionFreeSpeedModel::Agent{
                    .position = position,
                    .orientation = orientation,
                    .timeGap = timeGap,
                    .v0 = desiredSpeed,
                    .radius = radius,
                    .strengthNeighborRepulsion = strengthNeighborRepulsion,
                    .rangeNeighborRepulsion = rangeNeighborRepulsion,
                    .strengthGeometryRepulsion = strengthGeometryRepulsion,
                    .rangeGeometryRepulsion = rangeGeometryRepulsion};
            }),
            py::kw_only(),
            py::arg("position") = d.position,
            py::arg("orientation") = d.orientation,
            py::arg("time_gap") = d.timeGap,
            py::arg("desired_speed") = d.v0,
            py::arg("radius") = d.radius,
            py::arg("strength_neighbor_repulsion") = d.strengthNeighborRepulsion,
            py::arg("range_neighbor_repulsion") = d.rangeNeighborRepulsion,
            py::arg("strength_geometry_repulsion") = d.strengthGeometryRepulsion,
            py::arg("range_geometry_repulsion") = d.rangeGeometryRepulsion)
        .def_readwrite("position", &CollisionFreeSpeedModel::Agent::position)
        .def_readwrite("orientation", &CollisionFreeSpeedModel::Agent::orientation)
        .def_readwrite("time_gap", &CollisionFreeSpeedModel::Agent::timeGap)
        .def_readwrite("desired_speed", &CollisionFreeSpeedModel::Agent::v0)
        .def_readwrite("radius", &CollisionFreeSpeedModel::Agent::radius)
        .def_readwrite(
            "strength_neighbor_repulsion",
            &CollisionFreeSpeedModel::Agent::strengthNeighborRepulsion)
        .def_readwrite(
            "range_neighbor_repulsion", &CollisionFreeSpeedModel::Agent::rangeNeighborRepulsion)
        .def_readwrite(
            "strength_geometry_repulsion",
            &CollisionFreeSpeedModel::Agent::strengthGeometryRepulsion)
        .def_readwrite(
            "range_geometry_repulsion", &CollisionFreeSpeedModel::Agent::rangeGeometryRepulsion);
}
