// SPDX-License-Identifier: LGPL-3.0-or-later
#include "OperationalModel.hpp"
#include "SocialForceModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

void init_social_force_model(py::module_& m)
{
    py::class_<SocialForceModel, OperationalModel, py::smart_holder>(m, "SocialForceModel")
        .def(py::init<>());
    const SocialForceModel::Agent d{};
    py::class_<SocialForceModel::Agent>(m, "SocialForceModelState")
        .def(
            py::init([](Point position,
                        Point velocity,
                        double mass,
                        double desiredSpeed,
                        double reactionTime,
                        double agentScale,
                        double obstacleScale,
                        double forceDistance,
                        double radius,
                        double bodyForce,
                        double friction) {
                return SocialForceModel::Agent{
                    .position = position,
                    .velocity = velocity,
                    .mass = mass,
                    .desiredSpeed = desiredSpeed,
                    .reactionTime = reactionTime,
                    .agentScale = agentScale,
                    .obstacleScale = obstacleScale,
                    .forceDistance = forceDistance,
                    .radius = radius,
                    .bodyForce = bodyForce,
                    .friction = friction};
            }),
            py::kw_only(),
            py::arg("position") = d.position,
            py::arg("velocity") = d.velocity,
            py::arg("mass") = d.mass,
            py::arg("desired_speed") = d.desiredSpeed,
            py::arg("reaction_time") = d.reactionTime,
            py::arg("agent_scale") = d.agentScale,
            py::arg("obstacle_scale") = d.obstacleScale,
            py::arg("force_distance") = d.forceDistance,
            py::arg("radius") = d.radius,
            py::arg("body_force") = d.bodyForce,
            py::arg("friction") = d.friction)
        .def_property_readonly(
            "orientation",
            [](const SocialForceModel::Agent& obj) { return obj.velocity.Normalized(); })
        .def_readwrite("position", &SocialForceModel::Agent::position)
        .def_readwrite("velocity", &SocialForceModel::Agent::velocity)
        .def_readwrite("mass", &SocialForceModel::Agent::mass)
        .def_readwrite("desired_speed", &SocialForceModel::Agent::desiredSpeed)
        .def_readwrite("reaction_time", &SocialForceModel::Agent::reactionTime)
        .def_readwrite("agent_scale", &SocialForceModel::Agent::agentScale)
        .def_readwrite("obstacle_scale", &SocialForceModel::Agent::obstacleScale)
        .def_readwrite("force_distance", &SocialForceModel::Agent::forceDistance)
        .def_readwrite("radius", &SocialForceModel::Agent::radius)
        .def_readwrite("body_force", &SocialForceModel::Agent::bodyForce)
        .def_readwrite("friction", &SocialForceModel::Agent::friction);
}
