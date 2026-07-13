// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentRouting.hpp"
#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/CollisionFreeSpeedModel/CollisionFreeSpeedModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

using D = CollisionFreeSpeedModel::Defaults;

void init_collision_free_speed_model(py::module_& m)
{
    py::class_<CollisionFreeSpeedModel, OperationalModel, py::smart_holder>(
        m, "CollisionFreeSpeedModel")
        .def(py::init<>())
        .def(
            "compute_next",
            [](const CollisionFreeSpeedModel& self,
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
            "_compute_next_state",
            [](const CollisionFreeSpeedModel& self,
               double dt,
               const AgentState& current,
               const AgentRouting& routing,
               const CollisionGeometry& geometry,
               const NeighborhoodSearch<GenericAgent>& ns) -> AgentState {
                AgentState next = current;
                self.ComputeNext(dt, current, next, routing, geometry, ns);
                return next;
            },
            py::arg("dt"),
            py::arg("current"),
            py::arg("routing"),
            py::arg("geometry"),
            py::arg("neighborhood_search"))
        .def(
            "check_model_constraint",
            [](const CollisionFreeSpeedModel& self,
               const GenericAgent& agent,
               const NeighborhoodSearch<GenericAgent>& ns,
               const CollisionGeometry& geometry) {
                self.CheckModelConstraint(agent, ns, geometry);
            },
            py::arg("agent"),
            py::arg("neighborhood_search"),
            py::arg("geometry"));

    m.def(
        "CollisionFreeSpeedModelState",
        [](Point position,
           Point orientation,
           double timeGap,
           double desiredSpeed,
           double radius,
           double strengthNeighborRepulsion,
           double rangeNeighborRepulsion,
           double strengthGeometryRepulsion,
           double rangeGeometryRepulsion) -> AgentState {
            AgentState state = CollisionFreeSpeedModel::MakeState(position);
            state.orientation = orientation;
            state.timeGap = timeGap;
            state.v0 = desiredSpeed;
            state.radius = radius;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.rangeNeighborRepulsion = rangeNeighborRepulsion;
            state.strengthGeometryRepulsion = strengthGeometryRepulsion;
            state.rangeGeometryRepulsion = rangeGeometryRepulsion;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{},
        py::arg("time_gap") = D::timeGap,
        py::arg("desired_speed") = D::v0,
        py::arg("radius") = D::radius,
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("range_neighbor_repulsion") = D::rangeNeighborRepulsion,
        py::arg("strength_geometry_repulsion") = D::strengthGeometryRepulsion,
        py::arg("range_geometry_repulsion") = D::rangeGeometryRepulsion);
}
