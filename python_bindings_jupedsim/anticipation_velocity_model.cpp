// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "OperationalModels/AnticipationVelocityModel/AnticipationVelocityModel.hpp"
#include "OperationalModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <cstdint>

namespace py = pybind11;

using D = AnticipationVelocityModel::Defaults;

void init_anticipation_velocity_model(py::module_& m)
{
    py::class_<AnticipationVelocityModel, OperationalModel, py::smart_holder>(
        m, "AnticipationVelocityModel")
        .def(py::init<uint64_t>(), py::kw_only(), py::arg("rng_seed"));

    m.def(
        "AnticipationVelocityModelState",
        [](Point position,
           Point orientation,
           double strengthNeighborRepulsion,
           double rangeNeighborRepulsion,
           double wallBufferDistance,
           double anticipationTime,
           double reactionTime,
           Point velocity,
           double timeGap,
           double desiredSpeed,
           double radius,
           double pushoutStrength) -> AgentState {
            AgentState state = AnticipationVelocityModel::MakeState(position);
            state.orientation = orientation;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.rangeNeighborRepulsion = rangeNeighborRepulsion;
            state.reactionTime = reactionTime;
            state.velocity = velocity;
            state.timeGap = timeGap;
            state.v0 = desiredSpeed;
            state.radius = radius;
            auto& extras = std::get<AVMExtras>(*state.extras);
            extras.wallBufferDistance = wallBufferDistance;
            extras.anticipationTime = anticipationTime;
            extras.pushoutStrength = pushoutStrength;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{},
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("range_neighbor_repulsion") = D::rangeNeighborRepulsion,
        py::arg("wall_buffer_distance") = D::wallBufferDistance,
        py::arg("anticipation_time") = D::anticipationTime,
        py::arg("reaction_time") = D::reactionTime,
        py::arg("velocity") = Point{},
        py::arg("time_gap") = D::timeGap,
        py::arg("desired_speed") = D::v0,
        py::arg("radius") = D::radius,
        py::arg("pushout_strength") = D::pushoutStrength);
}
