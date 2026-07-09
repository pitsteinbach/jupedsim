// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/WarpDriver/WarpDriverModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <cstdint>

namespace py = pybind11;

using D = WarpDriverModel::Defaults;

void init_warp_driver_model(py::module_& m)
{
    py::class_<WarpDriverModel, OperationalModel, py::smart_holder>(m, "WarpDriverModel")
        .def(
            py::init<double, uint64_t>(),
            py::kw_only(),
            py::arg("sigma") = 0.3,
            py::arg("rng_seed") = 42);

    m.def(
        "WarpDriverModelState",
        [](Point position,
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
           int numSamples) -> AgentState {
            AgentState state = WarpDriverModel::MakeState(position);
            state.orientation = orientation;
            state.radius = radius;
            state.v0 = desiredSpeed;
            auto& extras = std::get<WarpExtras>(*state.extras);
            extras.stuckTime = stuckTime;
            extras.anchorX = anchorX;
            extras.anchorY = anchorY;
            extras.detourTime = detourTime;
            extras.detourSide = detourSide;
            extras.timeHorizon = timeHorizon;
            extras.stepSize = stepSize;
            extras.timeUncertainty = timeUncertainty;
            extras.velocityUncertaintyX = velocityUncertaintyX;
            extras.velocityUncertaintyY = velocityUncertaintyY;
            extras.numSamples = numSamples;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{},
        py::arg("radius") = D::radius,
        py::arg("desired_speed") = D::v0,
        py::arg("stuck_time") = 0.0,
        py::arg("anchor_x") = 0.0,
        py::arg("anchor_y") = 0.0,
        py::arg("detour_time") = 0.0,
        py::arg("detour_side") = 1,
        py::arg("time_horizon") = D::timeHorizon,
        py::arg("step_size") = D::stepSize,
        py::arg("time_uncertainty") = D::timeUncertainty,
        py::arg("velocity_uncertainty_x") = D::velocityUncertaintyX,
        py::arg("velocity_uncertainty_y") = D::velocityUncertaintyY,
        py::arg("num_samples") = D::numSamples);
}
