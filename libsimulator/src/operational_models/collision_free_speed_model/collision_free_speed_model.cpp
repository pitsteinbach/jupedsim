// SPDX-License-Identifier: LGPL-3.0-or-later
#include "collision_free_speed_model.hpp"

#include "agent_shape.hpp"
#include "agent_view.hpp"
#include "generic_agent.hpp"
#include "kernel.hpp"
#include "operational_model.hpp"
#include "operational_model_state.hpp"
#include "operational_model_type.hpp"
#include "simulation_error.hpp"
#include "visitor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace
{
/// How far to ask for walls. The repulsion decays with `range_geometry_repulsion` (0.02 m by
/// default), so a wall this far off contributes "nothing".
constexpr double wall_search_radius = 4.0;

/// Perimeter gap to a neighbor at @p relative_position from the agent.
double gap_to(
    const CollisionFreeSpeedModelState& agent,
    const CollisionFreeSpeedModelState& neighbor,
    const Vector& relative_position)
{
    return perimeter_distance(
        as_agent_shape(agent.shape),
        Point{},
        agent.orientation,
        as_agent_shape(neighbor.shape),
        at_offset(relative_position),
        neighbor.orientation);
}
} // namespace

CollisionFreeSpeedModel::CollisionFreeSpeedModel(
    double strength_neighbor_repulsion,
    double range_neighbor_repulsion,
    double strength_geometry_repulsion,
    double range_geometry_repulsion)
    : _strength_neighbor_repulsion(strength_neighbor_repulsion)
    , _range_neighbor_repulsion(range_neighbor_repulsion)
    , _strength_geometry_repulsion(strength_geometry_repulsion)
    , _range_geometry_repulsion(range_geometry_repulsion)
{
}

OperationalModelType CollisionFreeSpeedModel::type() const
{
    return OperationalModelType::CollisionFreeSpeed;
}

Vector CollisionFreeSpeedModel::compute_next_state(
    const OperationalModelState& current,
    OperationalModelState& next,
    const AgentStep& step) const
{
    const auto& current_state = std::get<State>(current);
    const auto neighborhood = step.other_agents_in_range(
        _cut_off_radius, [&step](const NeighborView& n) { return step.no_geometry_between(n); });

    Vector neighbor_repulsion{};
    for(const auto& neighbor : neighborhood) {
        neighbor_repulsion += this->neighbor_repulsion(current_state, neighbor);
    }

    Vector boundary_repulsion{};
    for(const auto& wall : step.walls_in_range(wall_search_radius)) {
        boundary_repulsion += this->boundary_repulsion(current_state, wall);
    }

    const auto desired_direction = step.route_orientation();
    auto direction = (desired_direction + neighbor_repulsion + boundary_repulsion).normalized();
    if(direction == Vector{}) {
        direction = current_state.orientation;
    }
    auto spacing = std::numeric_limits<double>::max();
    for(const auto& neighbor : neighborhood) {
        spacing = std::min(spacing, get_spacing(current_state, neighbor, direction));
    }

    const auto optimal_speed = this->optimal_speed(current_state, spacing, current_state.time_gap);
    const auto velocity = direction * optimal_speed;
    std::get<State>(next).orientation = direction;
    return velocity * step.dt();
}

void CollisionFreeSpeedModel::check_model_constraint(
    const GenericAgent& agent,
    const AgentView& view) const
{
    const auto& current_state = std::get<State>(agent.state);

    constexpr double r_max = 2.;
    std::visit(
        Overloaded{
            [](const CircleShape& circle) {
                constexpr double r_min = 0.;
                validate_constraint(circle.radius(), r_min, r_max, "radius", true);
            },
            [](const EllipseShape& ellipse) {
                constexpr double axis_min = 0.;
                validate_constraint(
                    ellipse.semi_axis_along(), axis_min, r_max, "semi_axis_along", true);
                validate_constraint(
                    ellipse.semi_axis_across(), axis_min, r_max, "semi_axis_across", true);
            }},
        current_state.shape);
    const auto& shape = as_agent_shape(current_state.shape);

    const auto v0 = current_state.v0;
    constexpr double v0_min = 0.;
    constexpr double v0_max = 10.;
    validate_constraint(v0, v0_min, v0_max, "v0");

    const auto time_gap = current_state.time_gap;
    constexpr double time_gap_min = 0.1;
    constexpr double time_gap_max = 10.;
    validate_constraint(time_gap, time_gap_min, time_gap_max, "timeGap");

    const auto neighbors = view.other_agents_in_range(2.0);
    for(const auto& neighbor : neighbors) {
        const auto& neighbor_state = std::get<State>(*neighbor.state);
        const auto distance = neighbor.relative_position.norm();
        if(gap_to(current_state, neighbor_state, neighbor.relative_position) <= 0) {
            throw SimulationError(
                "Model constraint violation: Agent {} too close to agent {}: distance {}",
                agent.location.xy(),
                agent.location.xy() + neighbor.relative_position,
                distance);
        }
    }

    for(const auto& wall : view.walls_in_range(r_max)) {
        const auto reach = shape.centre_to_perimeter(-wall.normal, current_state.orientation);
        if(wall.distance <= reach) {
            throw SimulationError(
                "Model constraint violation: Agent at {} too close to geometry boundaries, "
                "distance < {}",
                agent.location.xy(),
                reach);
        }
    }
}

double CollisionFreeSpeedModel::optimal_speed(
    const State& current_state,
    double spacing,
    double time_gap) const
{
    return std::min(std::max(spacing / time_gap, 0.0), current_state.v0);
}

double CollisionFreeSpeedModel::get_spacing(
    const State& current_state,
    const NeighborView& neighbor,
    const Vector& direction) const
{
    const auto& other = std::get<State>(*neighbor.state);
    const auto distp12 = neighbor.relative_position;
    const auto in_front = direction.scalar_product(distp12) >= 0;
    if(!in_front) {
        return std::numeric_limits<double>::max();
    }

    // Half-widths across the walking direction; exact for circles, an approximation otherwise.
    const auto left = direction.rotate90_deg();
    const auto half_width =
        as_agent_shape(current_state.shape).centre_to_perimeter(left, current_state.orientation) +
        as_agent_shape(other.shape).centre_to_perimeter(left, other.orientation);
    bool in_corridor = std::abs(left.scalar_product(distp12)) <= half_width;
    if(!in_corridor) {
        return std::numeric_limits<double>::max();
    }
    return gap_to(current_state, other, distp12);
}
Vector CollisionFreeSpeedModel::neighbor_repulsion(
    const State& current_state,
    const NeighborView& neighbor) const
{
    const auto& other = std::get<State>(*neighbor.state);
    const auto direction = neighbor.relative_position.normalized();
    const auto gap = gap_to(current_state, other, neighbor.relative_position);
    return direction *
           -(this->_strength_neighbor_repulsion * exp(-gap / this->_range_neighbor_repulsion));
}

Vector CollisionFreeSpeedModel::boundary_repulsion(
    const State& current_state,
    const WallView& boundary) const
{
    const auto l = as_agent_shape(current_state.shape)
                       .centre_to_perimeter(-boundary.normal, current_state.orientation);
    const auto r_iw =
        -_strength_geometry_repulsion * exp((l - boundary.distance) / _range_geometry_repulsion);
    return -boundary.normal * r_iw; // The repulsion points away from the agent
}
