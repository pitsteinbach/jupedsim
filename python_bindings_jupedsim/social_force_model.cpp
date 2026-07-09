// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "OperationalModels/SocialForceModel/SocialForceModel.hpp"
#include "OperationalModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

using D = SocialForceModel::Defaults;

void init_social_force_model(py::module_& m)
{
    py::class_<SocialForceModel, OperationalModel, py::smart_holder>(m, "SocialForceModel")
        .def(py::init<>());

    m.def(
        "SocialForceModelState",
        [](Point position,
           Point velocity,
           double mass,
           double desiredSpeed,
           double reactionTime,
           double agentScale,
           double obstacleScale,
           double forceDistance,
           double radius,
           double bodyForce,
           double friction) -> AgentState {
            AgentState state = SocialForceModel::MakeState(position);
            state.velocity = velocity;
            state.mass = mass;
            state.v0 = desiredSpeed;
            state.reactionTime = reactionTime;
            state.radius = radius;
            auto& extras = std::get<SFMExtras>(*state.extras);
            extras.agentScale = agentScale;
            extras.obstacleScale = obstacleScale;
            extras.forceDistance = forceDistance;
            extras.bodyForce = bodyForce;
            extras.friction = friction;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("velocity") = Point{},
        py::arg("mass") = D::mass,
        py::arg("desired_speed") = D::v0,
        py::arg("reaction_time") = D::reactionTime,
        py::arg("agent_scale") = D::agentScale,
        py::arg("obstacle_scale") = D::obstacleScale,
        py::arg("force_distance") = D::forceDistance,
        py::arg("radius") = D::radius,
        py::arg("body_force") = D::bodyForce,
        py::arg("friction") = D::friction);
}
