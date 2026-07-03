// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AnticipationVelocityModel.hpp"
#include "AnticipationVelocityModelBuilder.hpp"
#include "OperationalModel.hpp"
#include "conversion.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

void init_anticipation_velocity_model(py::module_& m)
{
    py::class_<AnticipationVelocityModel, OperationalModel, py::smart_holder>(
        m, "AnticipationVelocityModel");
    py::class_<AnticipationVelocityModelBuilder>(m, "AnticipationVelocityModelBuilder")
        .def(
            py::init<double, double>(),
            py::kw_only(),
            py::arg("pushout_strength"),
            py::arg("rng_seed"))
        .def("build", &AnticipationVelocityModelBuilder::Build);
    py::class_<AnticipationVelocityModel::Agent>(m, "AnticipationVelocityModelState")
        .def_static("_defaults", []() { return AnticipationVelocityModel::Agent{}; })
        .def(
            py::init([](std::tuple<double, double> orientation,
                        double strengthNeighborRepulsion,
                        double rangeNeighborRepulsion,
                        double wallBufferDistance,
                        double anticipationTime,
                        double reactionTime,
                        double timeGap,
                        double desiredSpeed,
                        double radius) {
                return AnticipationVelocityModel::Agent{
                    .orientation = intoPoint(orientation),
                    .strengthNeighborRepulsion = strengthNeighborRepulsion,
                    .rangeNeighborRepulsion = rangeNeighborRepulsion,
                    .wallBufferDistance = wallBufferDistance,
                    .anticipationTime = anticipationTime,
                    .reactionTime = reactionTime,
                    .timeGap = timeGap,
                    .v0 = desiredSpeed,
                    .radius = radius};
            }),
            py::kw_only(),
            py::arg("orientation"),
            py::arg("strength_neighbor_repulsion"),
            py::arg("range_neighbor_repulsion"),
            py::arg("wall_buffer_distance"),
            py::arg("anticipation_time"),
            py::arg("reaction_time"),
            py::arg("time_gap"),
            py::arg("desired_speed"),
            py::arg("radius"))
        .def_readwrite("orientation", &AnticipationVelocityModel::Agent::orientation)
        .def_readwrite(
            "strength_neighbor_repulsion",
            &AnticipationVelocityModel::Agent::strengthNeighborRepulsion)
        .def_readwrite(
            "range_neighbor_repulsion", &AnticipationVelocityModel::Agent::rangeNeighborRepulsion)
        .def_readwrite(
            "wall_buffer_distance", &AnticipationVelocityModel::Agent::wallBufferDistance)
        .def_readwrite("anticipation_time", &AnticipationVelocityModel::Agent::anticipationTime)
        .def_readwrite("reaction_time", &AnticipationVelocityModel::Agent::reactionTime)
        .def_readwrite("velocity", &AnticipationVelocityModel::Agent::velocity)
        .def_readwrite("time_gap", &AnticipationVelocityModel::Agent::timeGap)
        .def_readwrite("desired_speed", &AnticipationVelocityModel::Agent::v0)
        .def_readwrite("radius", &AnticipationVelocityModel::Agent::radius);
}
