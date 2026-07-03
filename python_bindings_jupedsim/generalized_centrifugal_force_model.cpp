// SPDX-License-Identifier: LGPL-3.0-or-later
#include "GeneralizedCentrifugalForceModel.hpp"
#include "GeneralizedCentrifugalForceModelBuilder.hpp"
#include "OperationalModel.hpp"
#include "conversion.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <tuple>

namespace py = pybind11;

void init_generalized_centrifugal_force_model(py::module_& m)
{
    py::class_<GeneralizedCentrifugalForceModel, OperationalModel, py::smart_holder>(
        m, "GeneralizedCentrifugalForceModel");
    py::class_<GeneralizedCentrifugalForceModelBuilder>(
        m, "GeneralizedCentrifugalForceModelBuilder")
        .def(
            py::init<double, double, double, double, double, double, double, double>(),
            py::kw_only(),
            py::arg("strength_neighbor_repulsion"),
            py::arg("strength_geometry_repulsion"),
            py::arg("max_neighbor_interaction_distance"),
            py::arg("max_geometry_interaction_distance"),
            py::arg("max_neighbor_interpolation_distance"),
            py::arg("max_geometry_interpolation_distance"),
            py::arg("max_neighbor_repulsion_force"),
            py::arg("max_geometry_repulsion_force"))
        .def("build", &GeneralizedCentrifugalForceModelBuilder::Build);
    py::class_<GeneralizedCentrifugalForceModel::Agent>(
        m, "GeneralizedCentrifugalForceModelState")
        .def_static("_defaults", []() { return GeneralizedCentrifugalForceModel::Agent{}; })
        .def(
            py::init([](std::tuple<double, double> orientation,
                        double speed,
                        std::tuple<double, double> desiredOrientation,
                        double mass,
                        double tau,
                        double desiredSpeed,
                        double av,
                        double amin,
                        double bmin,
                        double bmax) {
                return GeneralizedCentrifugalForceModel::Agent{
                    .orientation = intoPoint(orientation),
                    .speed = speed,
                    .e0 = intoPoint(desiredOrientation),
                    .mass = mass,
                    .tau = tau,
                    .v0 = desiredSpeed,
                    .Av = av,
                    .AMin = amin,
                    .BMin = bmin,
                    .BMax = bmax};
            }),
            py::kw_only(),
            py::arg("orientation"),
            py::arg("speed"),
            py::arg("desired_direction"),
            py::arg("mass"),
            py::arg("tau"),
            py::arg("desired_speed"),
            py::arg("a_v"),
            py::arg("a_min"),
            py::arg("b_min"),
            py::arg("b_max"))
        .def_readwrite("orientation", &GeneralizedCentrifugalForceModel::Agent::orientation)
        .def_readwrite("speed", &GeneralizedCentrifugalForceModel::Agent::speed)
        .def_property(
            "desired_direction",
            [](const GeneralizedCentrifugalForceModel::Agent& obj) { return intoTuple(obj.e0); },
            [](GeneralizedCentrifugalForceModel::Agent& obj, std::tuple<double, double> pt) {
                obj.e0 = intoPoint(pt);
            })
        .def_readwrite(
            "orientation_delay", &GeneralizedCentrifugalForceModel::Agent::orientationDelay)
        .def_readwrite("mass", &GeneralizedCentrifugalForceModel::Agent::mass)
        .def_readwrite("tau", &GeneralizedCentrifugalForceModel::Agent::tau)
        .def_readwrite("desired_speed", &GeneralizedCentrifugalForceModel::Agent::v0)
        .def_readwrite("a_v", &GeneralizedCentrifugalForceModel::Agent::Av)
        .def_readwrite("a_min", &GeneralizedCentrifugalForceModel::Agent::AMin)
        .def_readwrite("b_min", &GeneralizedCentrifugalForceModel::Agent::BMin)
        .def_readwrite("b_max", &GeneralizedCentrifugalForceModel::Agent::BMax);
}
