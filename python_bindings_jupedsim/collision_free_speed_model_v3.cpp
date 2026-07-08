// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "OperationalModels/CollisionFreeSpeedModelV3/CollisionFreeSpeedModelV3.hpp"
#include "OperationalModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

using D = CollisionFreeSpeedModelV3::Defaults;

void init_collision_free_speed_model_v3(py::module_& m)
{
    py::class_<CollisionFreeSpeedModelV3, OperationalModel, py::smart_holder>(
        m, "CollisionFreeSpeedModelV3")
        .def(py::init<>());

    m.def(
        "CollisionFreeSpeedModelV3State",
        [](Point position,
           Point orientation,
           double strengthNeighborRepulsion,
           double rangeNeighborRepulsion,
           double strengthGeometryRepulsion,
           double rangeGeometryRepulsion,
           double rangeXScale,
           double rangeYScale,
           double thetaMaxUpperBound,
           double agentBuffer,
           double timeGap,
           double desiredSpeed,
           double radius,
           double headingAngle) -> AgentState {
            AgentState state = CollisionFreeSpeedModelV3::MakeState(position);
            state.orientation = orientation;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.rangeNeighborRepulsion = rangeNeighborRepulsion;
            state.strengthGeometryRepulsion = strengthGeometryRepulsion;
            state.rangeGeometryRepulsion = rangeGeometryRepulsion;
            state.timeGap = timeGap;
            state.v0 = desiredSpeed;
            state.radius = radius;
            auto& extras = std::get<CFSv3Extras>(state.extras);
            extras.rangeXScale = rangeXScale;
            extras.rangeYScale = rangeYScale;
            extras.thetaMaxUpperBound = thetaMaxUpperBound;
            extras.agentBuffer = agentBuffer;
            extras.headingAngle = headingAngle;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{1.0, 0.0},
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("range_neighbor_repulsion") = D::rangeNeighborRepulsion,
        py::arg("strength_geometry_repulsion") = D::strengthGeometryRepulsion,
        py::arg("range_geometry_repulsion") = D::rangeGeometryRepulsion,
        py::arg("range_x_scale") = D::rangeXScale,
        py::arg("range_y_scale") = D::rangeYScale,
        py::arg("theta_max_upper_bound") = D::thetaMaxUpperBound,
        py::arg("agent_buffer") = D::agentBuffer,
        py::arg("time_gap") = D::timeGap,
        py::arg("desired_speed") = D::v0,
        py::arg("radius") = D::radius,
        py::arg("heading_angle") = D::headingAngle);
}
