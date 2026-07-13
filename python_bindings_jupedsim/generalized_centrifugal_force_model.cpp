// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/GeneralizedCentrifugalForceModel/GeneralizedCentrifugalForceModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

using D = GeneralizedCentrifugalForceModel::Defaults;

void init_generalized_centrifugal_force_model(py::module_& m)
{
    py::class_<GeneralizedCentrifugalForceModel, OperationalModel, py::smart_holder>(
        m, "GeneralizedCentrifugalForceModel")
        .def(py::init<>())
        .def(
            "compute_next",
            [](const GeneralizedCentrifugalForceModel& self,
               double dt,
               const GenericAgent& current,
               GenericAgent& next,
               const CollisionGeometry& geometry,
               const NeighborhoodSearch<GenericAgent>& ns) {
                self.ComputeNext(dt, current.model, next.model, current.routing, geometry, ns);
            },
            py::arg("dt"),
            py::arg("current"),
            py::arg("next"),
            py::arg("geometry"),
            py::arg("neighborhood_search"))
        .def(
            "check_model_constraint",
            [](const GeneralizedCentrifugalForceModel& self,
               const GenericAgent& agent,
               const NeighborhoodSearch<GenericAgent>& ns,
               const CollisionGeometry& geometry) {
                self.CheckModelConstraint(agent, ns, geometry);
            },
            py::arg("agent"),
            py::arg("neighborhood_search"),
            py::arg("geometry"));

    m.def(
        "GeneralizedCentrifugalForceModelState",
        [](Point position,
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
           double maxGeometryRepulsionForce) -> AgentState {
            AgentState state = GeneralizedCentrifugalForceModel::MakeState(position);
            state.orientation = orientation;
            state.mass = mass;
            state.reactionTime = tau;
            state.v0 = desiredSpeed;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.strengthGeometryRepulsion = strengthGeometryRepulsion;
            auto& extras = std::get<GCFMExtras>(*state.extras);
            extras.speed = speed;
            extras.e0 = desiredDirection;
            extras.orientationDelay = orientationDelay;
            extras.Av = av;
            extras.AMin = amin;
            extras.BMin = bmin;
            extras.BMax = bmax;
            extras.maxNeighborInteractionDistance = maxNeighborInteractionDistance;
            extras.maxGeometryInteractionDistance = maxGeometryInteractionDistance;
            extras.maxNeighborInterpolationDistance = maxNeighborInterpolationDistance;
            extras.maxGeometryInterpolationDistance = maxGeometryInterpolationDistance;
            extras.maxNeighborRepulsionForce = maxNeighborRepulsionForce;
            extras.maxGeometryRepulsionForce = maxGeometryRepulsionForce;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{1.0, 0.0},
        py::arg("speed") = 0.0,
        py::arg("desired_direction") = Point{},
        py::arg("orientation_delay") = 0,
        py::arg("mass") = D::mass,
        py::arg("tau") = D::reactionTime,
        py::arg("desired_speed") = D::v0,
        py::arg("a_v") = D::Av,
        py::arg("a_min") = D::AMin,
        py::arg("b_min") = D::BMin,
        py::arg("b_max") = D::BMax,
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("strength_geometry_repulsion") = D::strengthGeometryRepulsion,
        py::arg("max_neighbor_interaction_distance") = D::maxNeighborInteractionDistance,
        py::arg("max_geometry_interaction_distance") = D::maxGeometryInteractionDistance,
        py::arg("max_neighbor_interpolation_distance") = D::maxNeighborInterpolationDistance,
        py::arg("max_geometry_interpolation_distance") = D::maxGeometryInterpolationDistance,
        py::arg("max_neighbor_repulsion_force") = D::maxNeighborRepulsionForce,
        py::arg("max_geometry_repulsion_force") = D::maxGeometryRepulsionForce);
}
