// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "OperationalModels/CollisionFreeSpeedModel/CollisionFreeSpeedModel.hpp"
#include "OperationalModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

using D = CollisionFreeSpeedModel::Defaults;

void init_collision_free_speed_model(py::module_& m)
{
    py::class_<CollisionFreeSpeedModel, OperationalModel, py::smart_holder>(
        m, "CollisionFreeSpeedModel")
        .def(py::init<>());

    m.def(
        "CollisionFreeSpeedModelState",
        [](Point position,
           Point orientation,
           double timeGap,
           double desiredSpeed,
           double radius,
           double strengthNeighborRepulsion,
           double rangeNeighborRepulsion,
           double strengthGeometryRepulsion,
           double rangeGeometryRepulsion) -> AgentState {
            AgentState state = CollisionFreeSpeedModel::MakeState(position);
            state.orientation = orientation;
            state.timeGap = timeGap;
            state.v0 = desiredSpeed;
            state.radius = radius;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.rangeNeighborRepulsion = rangeNeighborRepulsion;
            state.strengthGeometryRepulsion = strengthGeometryRepulsion;
            state.rangeGeometryRepulsion = rangeGeometryRepulsion;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{},
        py::arg("time_gap") = D::timeGap,
        py::arg("desired_speed") = D::v0,
        py::arg("radius") = D::radius,
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("range_neighbor_repulsion") = D::rangeNeighborRepulsion,
        py::arg("strength_geometry_repulsion") = D::strengthGeometryRepulsion,
        py::arg("range_geometry_repulsion") = D::rangeGeometryRepulsion);
}
