// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentState.hpp"
#include "OperationalModels/OperationalModelType.hpp"
#include "type_casters.hpp" // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include <variant>

namespace py = pybind11;

// Helper: get reference to the specific extras type, raising AttributeError if
// the state is not using that model's extras.
template <typename E>
E& GetExtras(AgentState& s, const char* attrName)
{
    if(auto* e = std::get_if<E>(&s.extras)) {
        return *e;
    }
    throw py::attribute_error(
        std::string("'AgentState' has no attribute '") + attrName + "'");
}

template <typename E>
const E& GetExtras(const AgentState& s, const char* attrName)
{
    if(const auto* e = std::get_if<E>(&s.extras)) {
        return *e;
    }
    throw py::attribute_error(
        std::string("'AgentState' has no attribute '") + attrName + "'");
}

void init_agent_state(py::module_& m)
{
    // ---- Extras structs (for direct construction and type inspection) ----

    py::class_<GCFMExtras>(m, "GCFMExtras")
        .def(py::init<>())
        .def_readwrite("speed", &GCFMExtras::speed)
        .def_readwrite("desired_direction", &GCFMExtras::e0)
        .def_readwrite("orientation_delay", &GCFMExtras::orientationDelay)
        .def_readwrite("a_v", &GCFMExtras::Av)
        .def_readwrite("a_min", &GCFMExtras::AMin)
        .def_readwrite("b_min", &GCFMExtras::BMin)
        .def_readwrite("b_max", &GCFMExtras::BMax)
        .def_readwrite(
            "max_neighbor_interaction_distance",
            &GCFMExtras::maxNeighborInteractionDistance)
        .def_readwrite(
            "max_geometry_interaction_distance",
            &GCFMExtras::maxGeometryInteractionDistance)
        .def_readwrite(
            "max_neighbor_interpolation_distance",
            &GCFMExtras::maxNeighborInterpolationDistance)
        .def_readwrite(
            "max_geometry_interpolation_distance",
            &GCFMExtras::maxGeometryInterpolationDistance)
        .def_readwrite(
            "max_neighbor_repulsion_force", &GCFMExtras::maxNeighborRepulsionForce)
        .def_readwrite(
            "max_geometry_repulsion_force", &GCFMExtras::maxGeometryRepulsionForce);

    py::class_<SFMExtras>(m, "SFMExtras")
        .def(py::init<>())
        .def_readwrite("agent_scale", &SFMExtras::agentScale)
        .def_readwrite("obstacle_scale", &SFMExtras::obstacleScale)
        .def_readwrite("force_distance", &SFMExtras::forceDistance)
        .def_readwrite("body_force", &SFMExtras::bodyForce)
        .def_readwrite("friction", &SFMExtras::friction);

    py::class_<AVMExtras>(m, "AVMExtras")
        .def(py::init<>())
        .def_readwrite("wall_buffer_distance", &AVMExtras::wallBufferDistance)
        .def_readwrite("anticipation_time", &AVMExtras::anticipationTime)
        .def_readwrite("pushout_strength", &AVMExtras::pushoutStrength);

    py::class_<CFSv3Extras>(m, "CFSv3Extras")
        .def(py::init<>())
        .def_readwrite("range_x_scale", &CFSv3Extras::rangeXScale)
        .def_readwrite("range_y_scale", &CFSv3Extras::rangeYScale)
        .def_readwrite("theta_max_upper_bound", &CFSv3Extras::thetaMaxUpperBound)
        .def_readwrite("agent_buffer", &CFSv3Extras::agentBuffer)
        .def_readwrite("heading_angle", &CFSv3Extras::headingAngle);

    py::class_<WarpExtras>(m, "WarpExtras")
        .def(py::init<>())
        .def_readwrite("stuck_time", &WarpExtras::stuckTime)
        .def_readwrite("anchor_x", &WarpExtras::anchorX)
        .def_readwrite("anchor_y", &WarpExtras::anchorY)
        .def_readwrite("detour_time", &WarpExtras::detourTime)
        .def_readwrite("detour_side", &WarpExtras::detourSide)
        .def_readwrite("time_horizon", &WarpExtras::timeHorizon)
        .def_readwrite("step_size", &WarpExtras::stepSize)
        .def_readwrite("time_uncertainty", &WarpExtras::timeUncertainty)
        .def_readwrite("velocity_uncertainty_x", &WarpExtras::velocityUncertaintyX)
        .def_readwrite("velocity_uncertainty_y", &WarpExtras::velocityUncertaintyY)
        .def_readwrite("num_samples", &WarpExtras::numSamples);

    // ---- AgentState: unified state for all built-in models ----
    // Common fields use snake_case Python names.
    // Model-specific extras fields are also accessible directly on AgentState;
    // accessing a field that doesn't belong to the agent's active model type
    // raises AttributeError.

    py::class_<AgentState>(m, "AgentState")
        .def(py::init<>())
        .def_readwrite("type", &AgentState::type)
        .def_readwrite("position", &AgentState::position)
        // Common optional fields
        .def_readwrite("orientation", &AgentState::orientation)
        .def_readwrite("desired_speed", &AgentState::v0)
        .def_readwrite("radius", &AgentState::radius)
        .def_readwrite("time_gap", &AgentState::timeGap)
        .def_readwrite(
            "strength_neighbor_repulsion", &AgentState::strengthNeighborRepulsion)
        .def_readwrite(
            "range_neighbor_repulsion", &AgentState::rangeNeighborRepulsion)
        .def_readwrite(
            "strength_geometry_repulsion", &AgentState::strengthGeometryRepulsion)
        .def_readwrite(
            "range_geometry_repulsion", &AgentState::rangeGeometryRepulsion)
        .def_readwrite("velocity", &AgentState::velocity)
        .def_readwrite("mass", &AgentState::mass)
        // reaction_time is the canonical name; tau is the GCFM alias
        .def_readwrite("reaction_time", &AgentState::reactionTime)
        .def_property(
            "tau",
            [](const AgentState& s) { return s.reactionTime; },
            [](AgentState& s, std::optional<double> v) { s.reactionTime = v; })
        // Raw extras variant (for type inspection or bulk replacement)
        .def_property(
            "extras",
            [](AgentState& s) -> py::object {
                return std::visit(
                    [](auto& e) -> py::object {
                        using T = std::decay_t<decltype(e)>;
                        if constexpr(std::is_same_v<T, std::monostate>) {
                            return py::none();
                        } else {
                            return py::cast(&e);
                        }
                    },
                    s.extras);
            },
            [](AgentState& s, const py::object& obj) {
                if(obj.is_none()) {
                    s.extras = std::monostate{};
                } else if(py::isinstance<GCFMExtras>(obj)) {
                    s.extras = obj.cast<GCFMExtras>();
                } else if(py::isinstance<SFMExtras>(obj)) {
                    s.extras = obj.cast<SFMExtras>();
                } else if(py::isinstance<AVMExtras>(obj)) {
                    s.extras = obj.cast<AVMExtras>();
                } else if(py::isinstance<CFSv3Extras>(obj)) {
                    s.extras = obj.cast<CFSv3Extras>();
                } else if(py::isinstance<WarpExtras>(obj)) {
                    s.extras = obj.cast<WarpExtras>();
                } else {
                    throw py::type_error(
                        "extras must be GCFMExtras, SFMExtras, AVMExtras, "
                        "CFSv3Extras, WarpExtras, or None");
                }
            })
        // ---- GCFM-specific extras ----
        .def_property(
            "speed",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "speed").speed;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "speed").speed = v;
            })
        .def_property(
            "desired_direction",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "desired_direction").e0;
            },
            [](AgentState& s, Point v) {
                GetExtras<GCFMExtras>(s, "desired_direction").e0 = v;
            })
        .def_property(
            "orientation_delay",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "orientation_delay").orientationDelay;
            },
            [](AgentState& s, int v) {
                GetExtras<GCFMExtras>(s, "orientation_delay").orientationDelay = v;
            })
        .def_property(
            "a_v",
            [](const AgentState& s) { return GetExtras<GCFMExtras>(s, "a_v").Av; },
            [](AgentState& s, double v) { GetExtras<GCFMExtras>(s, "a_v").Av = v; })
        .def_property(
            "a_min",
            [](const AgentState& s) { return GetExtras<GCFMExtras>(s, "a_min").AMin; },
            [](AgentState& s, double v) { GetExtras<GCFMExtras>(s, "a_min").AMin = v; })
        .def_property(
            "b_min",
            [](const AgentState& s) { return GetExtras<GCFMExtras>(s, "b_min").BMin; },
            [](AgentState& s, double v) { GetExtras<GCFMExtras>(s, "b_min").BMin = v; })
        .def_property(
            "b_max",
            [](const AgentState& s) { return GetExtras<GCFMExtras>(s, "b_max").BMax; },
            [](AgentState& s, double v) { GetExtras<GCFMExtras>(s, "b_max").BMax = v; })
        .def_property(
            "max_neighbor_interaction_distance",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "max_neighbor_interaction_distance")
                    .maxNeighborInteractionDistance;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "max_neighbor_interaction_distance")
                    .maxNeighborInteractionDistance = v;
            })
        .def_property(
            "max_geometry_interaction_distance",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "max_geometry_interaction_distance")
                    .maxGeometryInteractionDistance;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "max_geometry_interaction_distance")
                    .maxGeometryInteractionDistance = v;
            })
        .def_property(
            "max_neighbor_interpolation_distance",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "max_neighbor_interpolation_distance")
                    .maxNeighborInterpolationDistance;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "max_neighbor_interpolation_distance")
                    .maxNeighborInterpolationDistance = v;
            })
        .def_property(
            "max_geometry_interpolation_distance",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "max_geometry_interpolation_distance")
                    .maxGeometryInterpolationDistance;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "max_geometry_interpolation_distance")
                    .maxGeometryInterpolationDistance = v;
            })
        .def_property(
            "max_neighbor_repulsion_force",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "max_neighbor_repulsion_force")
                    .maxNeighborRepulsionForce;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "max_neighbor_repulsion_force")
                    .maxNeighborRepulsionForce = v;
            })
        .def_property(
            "max_geometry_repulsion_force",
            [](const AgentState& s) {
                return GetExtras<GCFMExtras>(s, "max_geometry_repulsion_force")
                    .maxGeometryRepulsionForce;
            },
            [](AgentState& s, double v) {
                GetExtras<GCFMExtras>(s, "max_geometry_repulsion_force")
                    .maxGeometryRepulsionForce = v;
            })
        // ---- SFM-specific extras ----
        .def_property(
            "agent_scale",
            [](const AgentState& s) {
                return GetExtras<SFMExtras>(s, "agent_scale").agentScale;
            },
            [](AgentState& s, double v) {
                GetExtras<SFMExtras>(s, "agent_scale").agentScale = v;
            })
        .def_property(
            "obstacle_scale",
            [](const AgentState& s) {
                return GetExtras<SFMExtras>(s, "obstacle_scale").obstacleScale;
            },
            [](AgentState& s, double v) {
                GetExtras<SFMExtras>(s, "obstacle_scale").obstacleScale = v;
            })
        .def_property(
            "force_distance",
            [](const AgentState& s) {
                return GetExtras<SFMExtras>(s, "force_distance").forceDistance;
            },
            [](AgentState& s, double v) {
                GetExtras<SFMExtras>(s, "force_distance").forceDistance = v;
            })
        .def_property(
            "body_force",
            [](const AgentState& s) {
                return GetExtras<SFMExtras>(s, "body_force").bodyForce;
            },
            [](AgentState& s, double v) {
                GetExtras<SFMExtras>(s, "body_force").bodyForce = v;
            })
        .def_property(
            "friction",
            [](const AgentState& s) {
                return GetExtras<SFMExtras>(s, "friction").friction;
            },
            [](AgentState& s, double v) {
                GetExtras<SFMExtras>(s, "friction").friction = v;
            })
        // ---- AVM-specific extras ----
        .def_property(
            "wall_buffer_distance",
            [](const AgentState& s) {
                return GetExtras<AVMExtras>(s, "wall_buffer_distance").wallBufferDistance;
            },
            [](AgentState& s, double v) {
                GetExtras<AVMExtras>(s, "wall_buffer_distance").wallBufferDistance = v;
            })
        .def_property(
            "anticipation_time",
            [](const AgentState& s) {
                return GetExtras<AVMExtras>(s, "anticipation_time").anticipationTime;
            },
            [](AgentState& s, double v) {
                GetExtras<AVMExtras>(s, "anticipation_time").anticipationTime = v;
            })
        .def_property(
            "pushout_strength",
            [](const AgentState& s) {
                return GetExtras<AVMExtras>(s, "pushout_strength").pushoutStrength;
            },
            [](AgentState& s, double v) {
                GetExtras<AVMExtras>(s, "pushout_strength").pushoutStrength = v;
            })
        // ---- CFSv3-specific extras ----
        .def_property(
            "range_x_scale",
            [](const AgentState& s) {
                return GetExtras<CFSv3Extras>(s, "range_x_scale").rangeXScale;
            },
            [](AgentState& s, double v) {
                GetExtras<CFSv3Extras>(s, "range_x_scale").rangeXScale = v;
            })
        .def_property(
            "range_y_scale",
            [](const AgentState& s) {
                return GetExtras<CFSv3Extras>(s, "range_y_scale").rangeYScale;
            },
            [](AgentState& s, double v) {
                GetExtras<CFSv3Extras>(s, "range_y_scale").rangeYScale = v;
            })
        .def_property(
            "theta_max_upper_bound",
            [](const AgentState& s) {
                return GetExtras<CFSv3Extras>(s, "theta_max_upper_bound").thetaMaxUpperBound;
            },
            [](AgentState& s, double v) {
                GetExtras<CFSv3Extras>(s, "theta_max_upper_bound").thetaMaxUpperBound = v;
            })
        .def_property(
            "agent_buffer",
            [](const AgentState& s) {
                return GetExtras<CFSv3Extras>(s, "agent_buffer").agentBuffer;
            },
            [](AgentState& s, double v) {
                GetExtras<CFSv3Extras>(s, "agent_buffer").agentBuffer = v;
            })
        .def_property(
            "heading_angle",
            [](const AgentState& s) {
                return GetExtras<CFSv3Extras>(s, "heading_angle").headingAngle;
            },
            [](AgentState& s, double v) {
                GetExtras<CFSv3Extras>(s, "heading_angle").headingAngle = v;
            })
        // ---- WarpDriver-specific extras ----
        .def_property(
            "stuck_time",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "stuck_time").stuckTime;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "stuck_time").stuckTime = v;
            })
        .def_property(
            "anchor_x",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "anchor_x").anchorX;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "anchor_x").anchorX = v;
            })
        .def_property(
            "anchor_y",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "anchor_y").anchorY;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "anchor_y").anchorY = v;
            })
        .def_property(
            "detour_time",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "detour_time").detourTime;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "detour_time").detourTime = v;
            })
        .def_property(
            "detour_side",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "detour_side").detourSide;
            },
            [](AgentState& s, int v) {
                GetExtras<WarpExtras>(s, "detour_side").detourSide = v;
            })
        .def_property(
            "time_horizon",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "time_horizon").timeHorizon;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "time_horizon").timeHorizon = v;
            })
        .def_property(
            "step_size",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "step_size").stepSize;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "step_size").stepSize = v;
            })
        .def_property(
            "time_uncertainty",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "time_uncertainty").timeUncertainty;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "time_uncertainty").timeUncertainty = v;
            })
        .def_property(
            "velocity_uncertainty_x",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "velocity_uncertainty_x").velocityUncertaintyX;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "velocity_uncertainty_x").velocityUncertaintyX = v;
            })
        .def_property(
            "velocity_uncertainty_y",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "velocity_uncertainty_y").velocityUncertaintyY;
            },
            [](AgentState& s, double v) {
                GetExtras<WarpExtras>(s, "velocity_uncertainty_y").velocityUncertaintyY = v;
            })
        .def_property(
            "num_samples",
            [](const AgentState& s) {
                return GetExtras<WarpExtras>(s, "num_samples").numSamples;
            },
            [](AgentState& s, int v) {
                GetExtras<WarpExtras>(s, "num_samples").numSamples = v;
            });
}
