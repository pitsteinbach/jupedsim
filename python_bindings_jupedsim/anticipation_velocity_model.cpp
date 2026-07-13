// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/AnticipationVelocityModel/AnticipationVelocityModel.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <cstdint>

namespace py = pybind11;

using D = AnticipationVelocityModel::Defaults;

void init_anticipation_velocity_model(py::module_& m)
{
    py::class_<AnticipationVelocityModel, OperationalModel, py::smart_holder>(
        m, "AnticipationVelocityModel")
        .def(py::init<uint64_t>(), py::kw_only(), py::arg("rng_seed"))
        .def(
            "compute_next",
            [](const AnticipationVelocityModel& self,
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
            [](const AnticipationVelocityModel& self,
               const GenericAgent& agent,
               const NeighborhoodSearch<GenericAgent>& ns,
               const CollisionGeometry& geometry) {
                self.CheckModelConstraint(agent, ns, geometry);
            },
            py::arg("agent"),
            py::arg("neighborhood_search"),
            py::arg("geometry"));

    m.def(
        "AnticipationVelocityModelState",
        [](Point position,
           Point orientation,
           double strengthNeighborRepulsion,
           double rangeNeighborRepulsion,
           double wallBufferDistance,
           double anticipationTime,
           double reactionTime,
           Point velocity,
           double timeGap,
           double desiredSpeed,
           double radius,
           double pushoutStrength) -> AgentState {
            AgentState state = AnticipationVelocityModel::MakeState(position);
            state.orientation = orientation;
            state.strengthNeighborRepulsion = strengthNeighborRepulsion;
            state.rangeNeighborRepulsion = rangeNeighborRepulsion;
            state.reactionTime = reactionTime;
            state.velocity = velocity;
            state.timeGap = timeGap;
            state.v0 = desiredSpeed;
            state.radius = radius;
            auto& extras = std::get<AVMExtras>(*state.extras);
            extras.wallBufferDistance = wallBufferDistance;
            extras.anticipationTime = anticipationTime;
            extras.pushoutStrength = pushoutStrength;
            return state;
        },
        py::kw_only(),
        py::arg("position") = Point{},
        py::arg("orientation") = Point{},
        py::arg("strength_neighbor_repulsion") = D::strengthNeighborRepulsion,
        py::arg("range_neighbor_repulsion") = D::rangeNeighborRepulsion,
        py::arg("wall_buffer_distance") = D::wallBufferDistance,
        py::arg("anticipation_time") = D::anticipationTime,
        py::arg("reaction_time") = D::reactionTime,
        py::arg("velocity") = Point{},
        py::arg("time_gap") = D::timeGap,
        py::arg("desired_speed") = D::v0,
        py::arg("radius") = D::radius,
        py::arg("pushout_strength") = D::pushoutStrength);
}
