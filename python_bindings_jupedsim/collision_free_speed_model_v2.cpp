// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/CollisionFreeSpeedModelV2/CollisionFreeSpeedModelV2.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

namespace py = pybind11;

using D = CollisionFreeSpeedModelV2::Defaults;

void init_collision_free_speed_model_v2(py::module_& m)
{
    py::class_<CollisionFreeSpeedModelV2, OperationalModel, py::smart_holder>(
        m, "CollisionFreeSpeedModelV2")
        .def(py::init<>())
        .def(
            "compute_next",
            [](const CollisionFreeSpeedModelV2& self,
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
            [](const CollisionFreeSpeedModelV2& self,
               const GenericAgent& agent,
               const NeighborhoodSearch<GenericAgent>& ns,
               const CollisionGeometry& geometry) {
                self.CheckModelConstraint(agent, ns, geometry);
            },
            py::arg("agent"),
            py::arg("neighborhood_search"),
            py::arg("geometry"));

    m.def(
        "CollisionFreeSpeedModelV2State",
        [](Point position,
           Point orientation,
           double strengthNeighborRepulsion,
           double rangeNeighborRepulsion,
           double strengthGeometryRepulsion,
           double rangeGeometryRepulsion,
           double timeGap,
           double desiredSpeed,
           double radius) -> AgentState {
            AgentState state = CollisionFreeSpeedModelV2::MakeState(position);
            state.orientation = orientation;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.rangeNeighborRepulsion = rangeNeighborRepulsion;
            state.strengthGeometryRepulsion = strengthGeometryRepulsion;
            state.rangeGeometryRepulsion = rangeGeometryRepulsion;
            state.timeGap = timeGap;
            state.v0 = desiredSpeed;
            state.radius = radius;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{},
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("range_neighbor_repulsion") = D::rangeNeighborRepulsion,
        py::arg("strength_geometry_repulsion") = D::strengthGeometryRepulsion,
        py::arg("range_geometry_repulsion") = D::rangeGeometryRepulsion,
        py::arg("time_gap") = D::timeGap,
        py::arg("desired_speed") = D::v0,
        py::arg("radius") = D::radius);
}
