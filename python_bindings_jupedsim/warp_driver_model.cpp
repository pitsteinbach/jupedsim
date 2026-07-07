// SPDX-License-Identifier: LGPL-3.0-or-later
#include "OperationalModel.hpp"
#include "WarpDriverModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <cstdint>

namespace py = pybind11;

void init_warp_driver_model(py::module_& m)
{
    py::class_<WarpDriverModel, OperationalModel, py::smart_holder>(m, "WarpDriverModel")
        .def(
            py::init<double, uint64_t>(),
            py::kw_only(),
            py::arg("sigma") = 0.3,
            py::arg("rng_seed") = 42);
    const WarpDriverModel::Agent d{};
    py::class_<WarpDriverModel::Agent>(m, "WarpDriverModelState")
        .def(
            py::init([](Point position,
                        Point orientation,
                        double radius,
                        double desiredSpeed,
                        double stuckTime,
                        double anchorX,
                        double anchorY,
                        double detourTime,
                        int detourSide,
                        double timeHorizon,
                        double stepSize,
                        double timeUncertainty,
                        double velocityUncertaintyX,
                        double velocityUncertaintyY,
                        int numSamples) {
                return WarpDriverModel::Agent{
                    .position = position,
                    .orientation = orientation,
                    .radius = radius,
                    .v0 = desiredSpeed,
                    .stuckTime = stuckTime,
                    .anchorX = anchorX,
                    .anchorY = anchorY,
                    .detourTime = detourTime,
                    .detourSide = detourSide,
                    .timeHorizon = timeHorizon,
                    .stepSize = stepSize,
                    .timeUncertainty = timeUncertainty,
                    .velocityUncertaintyX = velocityUncertaintyX,
                    .velocityUncertaintyY = velocityUncertaintyY,
                    .numSamples = numSamples};
            }),
            py::kw_only(),
            py::arg("position") = d.position,
            py::arg("orientation") = d.orientation,
            py::arg("radius") = d.radius,
            py::arg("desired_speed") = d.v0,
            py::arg("stuck_time") = d.stuckTime,
            py::arg("anchor_x") = d.anchorX,
            py::arg("anchor_y") = d.anchorY,
            py::arg("detour_time") = d.detourTime,
            py::arg("detour_side") = d.detourSide,
            py::arg("time_horizon") = d.timeHorizon,
            py::arg("step_size") = d.stepSize,
            py::arg("time_uncertainty") = d.timeUncertainty,
            py::arg("velocity_uncertainty_x") = d.velocityUncertaintyX,
            py::arg("velocity_uncertainty_y") = d.velocityUncertaintyY,
            py::arg("num_samples") = d.numSamples)
        .def_readwrite("position", &WarpDriverModel::Agent::position)
        .def_readwrite("orientation", &WarpDriverModel::Agent::orientation)
        .def_readwrite("radius", &WarpDriverModel::Agent::radius)
        .def_readwrite("desired_speed", &WarpDriverModel::Agent::v0)
        .def_readwrite("stuck_time", &WarpDriverModel::Agent::stuckTime)
        .def_readwrite("anchor_x", &WarpDriverModel::Agent::anchorX)
        .def_readwrite("anchor_y", &WarpDriverModel::Agent::anchorY)
        .def_readwrite("detour_time", &WarpDriverModel::Agent::detourTime)
        .def_readwrite("detour_side", &WarpDriverModel::Agent::detourSide)
        .def_readwrite("time_horizon", &WarpDriverModel::Agent::timeHorizon)
        .def_readwrite("step_size", &WarpDriverModel::Agent::stepSize)
        .def_readwrite("time_uncertainty", &WarpDriverModel::Agent::timeUncertainty)
        .def_readwrite("velocity_uncertainty_x", &WarpDriverModel::Agent::velocityUncertaintyX)
        .def_readwrite("velocity_uncertainty_y", &WarpDriverModel::Agent::velocityUncertaintyY)
        .def_readwrite("num_samples", &WarpDriverModel::Agent::numSamples);
}
