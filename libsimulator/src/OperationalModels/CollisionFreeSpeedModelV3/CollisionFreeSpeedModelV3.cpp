// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CollisionFreeSpeedModelV3.hpp"

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "GeometricFunctions.hpp"
#include "LineSegment.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"
#include "SimulationError.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace
{
constexpr double Eps = 1e-6;
constexpr double SideEps = 0.05;
constexpr double SpacingBlendWeight = 0.15;
constexpr double TauTheta = 0.3;
constexpr double MinReverseSpeed = -0.01;

double NeighborInfluence(
    const std::vector<GenericAgent>& neighborhood,
    const Point& pos,
    const Point& reference_direction,
    const CFSv3Extras& extras,
    double strengthNeighborRepulsion,
    double rangeNeighborRepulsion)
{
    const auto range_x = std::max(Eps, rangeNeighborRepulsion * extras.rangeXScale);
    const auto range_y = std::max(Eps, rangeNeighborRepulsion * extras.rangeYScale);
    const auto theta_max = std::clamp(strengthNeighborRepulsion, 0.0, extras.thetaMaxUpperBound);

    double best_influence = 0.0;
    double best_weight = 0.0;
    for(const auto& neighbor : neighborhood) {
        const auto relative = Pos(neighbor) - pos;
        const auto x = reference_direction.ScalarProduct(relative);
        if(x <= 0.0) {
            continue;
        }
        const auto signed_lateral = reference_direction.CrossProduct(relative);
        const auto y = std::abs(signed_lateral);
        const auto longitudinal_weight = std::exp(-x / range_x);
        const auto lateral_weight = std::exp(-y / range_y);
        const auto weight = longitudinal_weight * lateral_weight;
        if(weight > best_weight) {
            best_weight = weight;
            best_influence = -weight * (signed_lateral / (std::abs(signed_lateral) + SideEps));
        }
    }
    return theta_max * std::tanh(best_influence);
}
} // namespace

AgentState CollisionFreeSpeedModelV3::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::COLLISION_FREE_SPEED_V3,
        .position = pos,
        .orientation = Point{1.0, 0.0},
        .v0 = Defaults::v0,
        .radius = Defaults::radius,
        .timeGap = Defaults::timeGap,
        .strengthNeighborRepulsion = Defaults::strengthNeighborRepulsion,
        .rangeNeighborRepulsion = Defaults::rangeNeighborRepulsion,
        .strengthGeometryRepulsion = Defaults::strengthGeometryRepulsion,
        .rangeGeometryRepulsion = Defaults::rangeGeometryRepulsion,
        .extras = CFSv3Extras{
            .rangeXScale = Defaults::rangeXScale,
            .rangeYScale = Defaults::rangeYScale,
            .thetaMaxUpperBound = Defaults::thetaMaxUpperBound,
            .agentBuffer = Defaults::agentBuffer,
            .headingAngle = Defaults::headingAngle,
        },
    };
}

OperationalModelType CollisionFreeSpeedModelV3::Type() const
{
    return OperationalModelType::COLLISION_FREE_SPEED_V3;
}

void CollisionFreeSpeedModelV3::ComputeNext(
    double dT,
    const GenericAgent& current,
    GenericAgent& next,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    auto neighborhood = neighborhoodSearch.GetNeighboringAgents(Pos(current), _cutOffRadius);
    const auto& boundary = geometry.LineSegmentsInApproxDistanceTo(Pos(current));

    std::erase_if(neighborhood, [&current, &boundary](const auto& neighbor) {
        if(current.id == neighbor.id) {
            return true;
        }
        const auto agent_to_neighbor = LineSegment(Pos(current), Pos(neighbor));
        return std::any_of(
            boundary.cbegin(), boundary.cend(), [&agent_to_neighbor](const auto& segment) {
                return intersects(agent_to_neighbor, segment);
            });
    });

    const auto boundaryRepulsion = std::accumulate(
        boundary.cbegin(),
        boundary.cend(),
        Point(0, 0),
        [this, &current](const auto& acc, const auto& element) {
            return acc + BoundaryRepulsion(current, element);
        });

    const auto& state = std::get<AgentState>(current.model);
    const auto& extras = std::get<CFSv3Extras>(*state.extras);
    const auto strengthN = state.strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion);
    const auto rangeN = state.rangeNeighborRepulsion.value_or(Defaults::rangeNeighborRepulsion);

    const auto desired_direction = (current.destination - Pos(current)).Normalized();
    auto reference_direction = (desired_direction + boundaryRepulsion).Normalized();
    if(reference_direction == Point{}) {
        reference_direction = state.orientation.value_or(Point{1.0, 0.0});
    }

    const auto heading_target =
        NeighborInfluence(neighborhood, Pos(current), reference_direction, extras, strengthN, rangeN);
    const auto alpha = std::clamp(dT / TauTheta, 0.0, 1.0);
    const auto heading_angle = extras.headingAngle + alpha * (heading_target - extras.headingAngle);
    auto direction =
        reference_direction.Rotate(std::cos(heading_angle), std::sin(heading_angle)).Normalized();
    if(direction == Point{}) {
        direction = reference_direction;
    }

    const auto spacing_move = std::accumulate(
        std::begin(neighborhood),
        std::end(neighborhood),
        std::numeric_limits<double>::max(),
        [&current, &direction, this](const auto& res, const auto& neighbor) {
            return std::min(res, GetSpacing(current, neighbor, direction));
        });

    const auto goal_direction =
        (desired_direction == Point{}) ? reference_direction : desired_direction;
    const auto spacing_goal = std::accumulate(
        std::begin(neighborhood),
        std::end(neighborhood),
        std::numeric_limits<double>::max(),
        [&current, &goal_direction, this](const auto& res, const auto& neighbor) {
            return std::min(res, GetSpacing(current, neighbor, goal_direction));
        });

    const auto spacing =
        spacing_move * (1.0 - SpacingBlendWeight) + spacing_goal * SpacingBlendWeight;

    const auto optimal_speed =
        OptimalSpeed(current, spacing, state.timeGap.value_or(Defaults::timeGap));
    const auto velocity = direction * optimal_speed;
    auto& nextState = std::get<AgentState>(next.model);
    nextState.position = Pos(current) + velocity * dT;
    nextState.orientation = direction;
    auto& nextExtras = std::get<CFSv3Extras>(*nextState.extras);
    nextExtras.headingAngle = heading_angle;
}

void CollisionFreeSpeedModelV3::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    const auto& state = std::get<AgentState>(agent.model);
    const auto& extras = std::get<CFSv3Extras>(*state.extras);

    validateConstraint(state.radius.value_or(Defaults::radius), 0.0, 2.0, "radius", true);
    validateConstraint(state.v0.value_or(Defaults::v0), 0.0, 10.0, "v0");
    validateConstraint(state.timeGap.value_or(Defaults::timeGap), 0.1, 10.0, "timeGap");
    validateConstraint(
        state.strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion),
        0.0,
        std::numeric_limits<double>::max(),
        "strengthNeighborRepulsion");
    validateConstraint(
        state.rangeNeighborRepulsion.value_or(Defaults::rangeNeighborRepulsion),
        0.01,
        std::numeric_limits<double>::max(),
        "rangeNeighborRepulsion");
    validateConstraint(
        state.strengthGeometryRepulsion.value_or(Defaults::strengthGeometryRepulsion),
        0.0,
        std::numeric_limits<double>::max(),
        "strengthGeometryRepulsion");
    validateConstraint(
        state.rangeGeometryRepulsion.value_or(Defaults::rangeGeometryRepulsion),
        0.01,
        std::numeric_limits<double>::max(),
        "rangeGeometryRepulsion");
    validateConstraint(extras.rangeXScale, 0.01, std::numeric_limits<double>::max(), "rangeXScale");
    validateConstraint(extras.rangeYScale, 0.01, std::numeric_limits<double>::max(), "rangeYScale");
    validateConstraint(extras.thetaMaxUpperBound, 0.0, std::acos(-1.0), "thetaMaxUpperBound");
    validateConstraint(extras.agentBuffer, 0.0, 100.0, "agentBuffer");

    const auto r = state.radius.value_or(Defaults::radius);
    const auto neighbors = neighborhoodSearch.GetNeighboringAgents(Pos(agent), 2);
    for(const auto& neighbor : neighbors) {
        if(agent.id == neighbor.id) {
            continue;
        }
        const auto* nbState = std::get_if<AgentState>(&neighbor.model);
        if(!nbState) {
            continue;
        }
        const auto contactDist = r + nbState->radius.value_or(Defaults::radius);
        const auto distance = (Pos(agent) - Pos(neighbor)).Norm();
        if(contactDist >= distance) {
            throw SimulationError(
                "Model constraint violation: Agent {} too close to agent {}: distance {}",
                Pos(agent),
                Pos(neighbor),
                distance);
        }
    }

    const auto lineSegments = geometry.LineSegmentsInDistanceTo(r, Pos(agent));
    if(std::begin(lineSegments) != std::end(lineSegments)) {
        throw SimulationError(
            "Model constraint violation: Agent {} too close to geometry boundaries, distance "
            "<= {}",
            Pos(agent),
            r);
    }
}

double CollisionFreeSpeedModelV3::OptimalSpeed(
    const GenericAgent& ped,
    double spacing,
    double time_gap) const
{
    const auto& state = std::get<AgentState>(ped.model);
    const auto& extras = std::get<CFSv3Extras>(*state.extras);
    const auto effective_spacing = spacing - extras.agentBuffer;
    return std::min(
        std::max(effective_spacing / time_gap, MinReverseSpeed),
        state.v0.value_or(Defaults::v0));
}

double CollisionFreeSpeedModelV3::GetSpacing(
    const GenericAgent& ped1,
    const GenericAgent& ped2,
    const Point& direction) const
{
    const auto* s1 = std::get_if<AgentState>(&ped1.model);
    const auto* s2 = std::get_if<AgentState>(&ped2.model);
    if(!s1 || !s2) {
        return std::numeric_limits<double>::max();
    }
    const auto distp12 = Pos(ped2) - Pos(ped1);
    if(direction.ScalarProduct(distp12) < 0.0) {
        return std::numeric_limits<double>::max();
    }
    const auto left = direction.Rotate90Deg();
    const auto l = s1->radius.value_or(Defaults::radius) + s2->radius.value_or(Defaults::radius);
    if(std::abs(left.ScalarProduct(distp12)) > l) {
        return std::numeric_limits<double>::max();
    }
    return distp12.Norm() - l;
}

Point CollisionFreeSpeedModelV3::BoundaryRepulsion(
    const GenericAgent& ped,
    const LineSegment& boundary_segment) const
{
    const auto pt = boundary_segment.ShortestPoint(Pos(ped));
    const auto dist_vec = pt - Pos(ped);
    const auto [dist, e_iw] = dist_vec.NormAndNormalized();
    const auto& state = std::get<AgentState>(ped.model);
    const auto l = state.radius.value_or(Defaults::radius);
    const auto strengthG = state.strengthGeometryRepulsion.value_or(Defaults::strengthGeometryRepulsion);
    const auto rangeG = state.rangeGeometryRepulsion.value_or(Defaults::rangeGeometryRepulsion);
    return e_iw * (-strengthG * std::exp((l - dist) / rangeG));
}
