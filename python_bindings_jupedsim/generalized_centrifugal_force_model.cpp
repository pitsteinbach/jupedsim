// SPDX-License-Identifier: LGPL-3.0-or-later
#include "GeneralizedCentrifugalForceModel.hpp"
#include "OperationalModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

void init_generalized_centrifugal_force_model(py::module_& m)
{
    py::class_<GeneralizedCentrifugalForceModel, OperationalModel, py::smart_holder>(
        m, "GeneralizedCentrifugalForceModel")
        .def(py::init<>());
    const GeneralizedCentrifugalForceModel::Agent d{};
    py::class_<GeneralizedCentrifugalForceModel::Agent>(m, "GeneralizedCentrifugalForceModelState")
        .def(
            py::init([](Point position,
                        Point orientation,
                        double speed,
                        Point desiredDirection,
                        int orientationDelay,
                        double mass,
                        double tau,
                        double desiredSpeed,
                        double av,
                        double amin,
                        double bmin,
                        double bmax,
                        double strengthNeighborRepulsion,
                        double strengthGeometryRepulsion,
                        double maxNeighborInteractionDistance,
                        double maxGeometryInteractionDistance,
                        double maxNeighborInterpolationDistance,
                        double maxGeometryInterpolationDistance,
                        double maxNeighborRepulsionForce,
                        double maxGeometryRepulsionForce) {
                return GeneralizedCentrifugalForceModel::Agent{
                    .position = position,
                    .orientation = orientation,
                    .speed = speed,
                    .e0 = desiredDirection,
                    .orientationDelay = orientationDelay,
                    .mass = mass,
                    .tau = tau,
                    .v0 = desiredSpeed,
                    .Av = av,
                    .AMin = amin,
                    .BMin = bmin,
                    .BMax = bmax,
                    .strengthNeighborRepulsion = strengthNeighborRepulsion,
                    .strengthGeometryRepulsion = strengthGeometryRepulsion,
                    .maxNeighborInteractionDistance = maxNeighborInteractionDistance,
                    .maxGeometryInteractionDistance = maxGeometryInteractionDistance,
                    .maxNeighborInterpolationDistance = maxNeighborInterpolationDistance,
                    .maxGeometryInterpolationDistance = maxGeometryInterpolationDistance,
                    .maxNeighborRepulsionForce = maxNeighborRepulsionForce,
                    .maxGeometryRepulsionForce = maxGeometryRepulsionForce};
            }),
            py::kw_only(),
            py::arg("position") = d.position,
            py::arg("orientation") = d.orientation,
            py::arg("speed") = d.speed,
            py::arg("desired_direction") = d.e0,
            py::arg("orientation_delay") = d.orientationDelay,
            py::arg("mass") = d.mass,
            py::arg("tau") = d.tau,
            py::arg("desired_speed") = d.v0,
            py::arg("a_v") = d.Av,
            py::arg("a_min") = d.AMin,
            py::arg("b_min") = d.BMin,
            py::arg("b_max") = d.BMax,
            py::arg("strength_neighbor_repulsion") = d.strengthNeighborRepulsion,
            py::arg("strength_geometry_repulsion") = d.strengthGeometryRepulsion,
            py::arg("max_neighbor_interaction_distance") = d.maxNeighborInteractionDistance,
            py::arg("max_geometry_interaction_distance") = d.maxGeometryInteractionDistance,
            py::arg("max_neighbor_interpolation_distance") = d.maxNeighborInterpolationDistance,
            py::arg("max_geometry_interpolation_distance") = d.maxGeometryInterpolationDistance,
            py::arg("max_neighbor_repulsion_force") = d.maxNeighborRepulsionForce,
            py::arg("max_geometry_repulsion_force") = d.maxGeometryRepulsionForce)
        .def_readwrite("position", &GeneralizedCentrifugalForceModel::Agent::position)
        .def_readwrite("orientation", &GeneralizedCentrifugalForceModel::Agent::orientation)
        .def_readwrite("speed", &GeneralizedCentrifugalForceModel::Agent::speed)
        .def_readwrite("desired_direction", &GeneralizedCentrifugalForceModel::Agent::e0)
        .def_readwrite(
            "orientation_delay", &GeneralizedCentrifugalForceModel::Agent::orientationDelay)
        .def_readwrite("mass", &GeneralizedCentrifugalForceModel::Agent::mass)
        .def_readwrite("tau", &GeneralizedCentrifugalForceModel::Agent::tau)
        .def_readwrite("desired_speed", &GeneralizedCentrifugalForceModel::Agent::v0)
        .def_readwrite("a_v", &GeneralizedCentrifugalForceModel::Agent::Av)
        .def_readwrite("a_min", &GeneralizedCentrifugalForceModel::Agent::AMin)
        .def_readwrite("b_min", &GeneralizedCentrifugalForceModel::Agent::BMin)
        .def_readwrite("b_max", &GeneralizedCentrifugalForceModel::Agent::BMax)
        .def_readwrite(
            "strength_neighbor_repulsion",
            &GeneralizedCentrifugalForceModel::Agent::strengthNeighborRepulsion)
        .def_readwrite(
            "strength_geometry_repulsion",
            &GeneralizedCentrifugalForceModel::Agent::strengthGeometryRepulsion)
        .def_readwrite(
            "max_neighbor_interaction_distance",
            &GeneralizedCentrifugalForceModel::Agent::maxNeighborInteractionDistance)
        .def_readwrite(
            "max_geometry_interaction_distance",
            &GeneralizedCentrifugalForceModel::Agent::maxGeometryInteractionDistance)
        .def_readwrite(
            "max_neighbor_interpolation_distance",
            &GeneralizedCentrifugalForceModel::Agent::maxNeighborInterpolationDistance)
        .def_readwrite(
            "max_geometry_interpolation_distance",
            &GeneralizedCentrifugalForceModel::Agent::maxGeometryInterpolationDistance)
        .def_readwrite(
            "max_neighbor_repulsion_force",
            &GeneralizedCentrifugalForceModel::Agent::maxNeighborRepulsionForce)
        .def_readwrite(
            "max_geometry_repulsion_force",
            &GeneralizedCentrifugalForceModel::Agent::maxGeometryRepulsionForce);
}
