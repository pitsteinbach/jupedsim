// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AnticipationVelocityModel.hpp"

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "GeometricFunctions.hpp"
#include "LineSegment.hpp"
#include "Macros.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"
#include "SimulationError.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

AnticipationVelocityModel::AnticipationVelocityModel(uint64_t rng_seed) : gen(rng_seed)
{
}

AgentState AnticipationVelocityModel::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::ANTICIPATION_VELOCITY_MODEL,
        .position = pos,
        .orientation = Point{0.0, 0.0},
        .v0 = Defaults::v0,
        .radius = Defaults::radius,
        .timeGap = Defaults::timeGap,
        .strengthNeighborRepulsion = Defaults::strengthNeighborRepulsion,
        .rangeNeighborRepulsion = Defaults::rangeNeighborRepulsion,
        .velocity = Point{0.0, 0.0},
        .reactionTime = Defaults::reactionTime,
        .extras =
            AVMExtras{
                .wallBufferDistance = Defaults::wallBufferDistance,
                .anticipationTime = Defaults::anticipationTime,
                .pushoutStrength = Defaults::pushoutStrength,
            },
    };
}

OperationalModelType AnticipationVelocityModel::Type() const
{
    return OperationalModelType::ANTICIPATION_VELOCITY_MODEL;
}

void AnticipationVelocityModel::ComputeNext(
    double dT,
    const AgentState& current,
    AgentState& next,
    const AgentRouting& routing,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    auto neighborhood = neighborhoodSearch.GetNeighboringAgents(current.position, _cutOffRadius);
    const auto& boundary = geometry.LineSegmentsInApproxDistanceTo(current.position);

    neighborhood.erase(
        std::remove_if(
            std::begin(neighborhood),
            std::end(neighborhood),
            [&current, &boundary](const auto& neighbor) {
                if(Pos(neighbor) == current.position) {
                    return true;
                }
                const auto agent_to_neighbor = LineSegment(current.position, Pos(neighbor));
                return std::find_if(
                           boundary.cbegin(),
                           boundary.cend(),
                           [&agent_to_neighbor](const auto& boundary_segment) {
                               return intersects(agent_to_neighbor, boundary_segment);
                           }) != boundary.end();
            }),
        std::end(neighborhood));

    const auto neighborRepulsion = std::accumulate(
        std::begin(neighborhood),
        std::end(neighborhood),
        Point{},
        [&current, &routing, this](const auto& res, const auto& neighbor) {
            return res + NeighborRepulsion(current, neighbor.model, routing);
        });

    const auto& extras = std::get<AVMExtras>(*current.extras);

    const auto desiredDirection = (routing.destination - current.position).Normalized();
    auto direction = (desiredDirection + neighborRepulsion).Normalized();
    if(direction == Point{}) {
        direction = current.orientation.value_or(Point{0.0, 0.0});
    }

    direction = UpdateDirection(current, direction, dT, routing);
    const auto spacing = std::accumulate(
        std::begin(neighborhood),
        std::end(neighborhood),
        std::numeric_limits<double>::max(),
        [&current, &direction, this](const auto& res, const auto& neighbor) {
            return std::min(res, GetSpacing(current, neighbor.model, direction));
        });

    const auto optimal_speed =
        OptimalSpeed(current, spacing, current.timeGap.value_or(Defaults::timeGap));
    direction = HandleWallAvoidance(
        direction,
        current.position,
        current.radius.value_or(Defaults::radius),
        boundary,
        extras.wallBufferDistance,
        extras.pushoutStrength);

    const auto velocity = direction * optimal_speed;
    next.position = current.position + velocity * dT;
    next.orientation = direction;
    next.velocity = velocity;
}

Point AnticipationVelocityModel::UpdateDirection(
    const AgentState& ped,
    const Point& calculatedDirection,
    double dt,
    const AgentRouting& routing) const
{
    const auto reactionTime = ped.reactionTime.value_or(Defaults::reactionTime);
    const Point desiredDirection = (routing.destination - ped.position).Normalized();
    const Point actualDirection = ped.orientation.value_or(Point{0.0, 0.0});
    Point updatedDirection;

    if(desiredDirection.ScalarProduct(calculatedDirection) *
           desiredDirection.ScalarProduct(actualDirection) <
       0) {
        updatedDirection = calculatedDirection;
    } else {
        const Point directionDerivative =
            (calculatedDirection.Normalized() - actualDirection) / reactionTime;
        updatedDirection = actualDirection + directionDerivative * dt;
    }

    return updatedDirection.Normalized();
}

void AnticipationVelocityModel::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    const auto& state = agent.model;
    const auto& extras = std::get<AVMExtras>(*state.extras);

    const auto r = state.radius.value_or(Defaults::radius);
    validateConstraint(r, 0.0, 2.0, "radius", true);
    validateConstraint(
        state.strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion),
        0.0,
        20.0,
        "strengthNeighborRepulsion");
    validateConstraint(
        state.rangeNeighborRepulsion.value_or(Defaults::rangeNeighborRepulsion),
        0.0,
        5.0,
        "rangeNeighborRepulsion",
        true);
    validateConstraint(extras.wallBufferDistance, 0.0, 1.0, "wallBufferDistance");
    validateConstraint(state.v0.value_or(Defaults::v0), 0.0, 10.0, "v0");
    validateConstraint(state.timeGap.value_or(Defaults::timeGap), 0.0, 10.0, "timeGap", true);
    validateConstraint(extras.anticipationTime, 0.0, 5.0, "anticipationTime");
    validateConstraint(
        state.reactionTime.value_or(Defaults::reactionTime), 0.0, 1.0, "reactionTime", true);

    const auto neighbors = neighborhoodSearch.GetNeighboringAgents(Pos(agent), 2);
    for(const auto& neighbor : neighbors) {
        if(agent.id == neighbor.id) {
            continue;
        }
        const auto& nbState = neighbor.model;
        const auto contactDist = r + nbState.radius.value_or(Defaults::radius);
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

double AnticipationVelocityModel::OptimalSpeed(
    const AgentState& ped,
    double spacing,
    double time_gap) const
{
    constexpr double creep_speed = 0.01;
    double speed = spacing / time_gap;
    if(std::abs(speed) < creep_speed) {
        const auto r = gen() % 3;
        speed = (r == 0) ? creep_speed : (r == 1) ? -creep_speed : 0.0;
    }
    return std::min(std::max(speed, -creep_speed), ped.v0.value_or(Defaults::v0));
}

double AnticipationVelocityModel::GetSpacing(
    const AgentState& ped1,
    const AgentState& ped2,
    const Point& direction) const
{
    const auto distp12 = ped2.position - ped1.position;
    if(direction.ScalarProduct(distp12) < 0) {
        return std::numeric_limits<double>::max();
    }
    const auto left = direction.Rotate90Deg();
    constexpr double buffer = 0.02;
    const auto l =
        ped1.radius.value_or(Defaults::radius) + ped2.radius.value_or(Defaults::radius) + buffer;
    if(std::abs(left.ScalarProduct(distp12)) > l) {
        return std::numeric_limits<double>::max();
    }
    return distp12.Norm() - l;
}

Point AnticipationVelocityModel::CalculateInfluenceDirection(
    const Point& desiredDirection,
    const Point& predictedDirection) const
{
    const Point orthogonalDirection = Point(-desiredDirection.y, desiredDirection.x).Normalized();
    const double alignment = orthogonalDirection.ScalarProduct(predictedDirection);
    Point influenceDirection = orthogonalDirection;
    if(fabs(alignment) < J_EPS) {
        if(gen() % 2 == 0) {
            influenceDirection = -orthogonalDirection;
        }
    } else if(alignment > 0) {
        influenceDirection = -orthogonalDirection;
    }
    return influenceDirection;
}

Point AnticipationVelocityModel::NeighborRepulsion(
    const AgentState& ped1,
    const AgentState& ped2,
    const AgentRouting& routing1) const
{
    const auto* e1 = ped1.extras ? std::get_if<AVMExtras>(&*ped1.extras) : nullptr;
    const auto* e2 = ped2.extras ? std::get_if<AVMExtras>(&*ped2.extras) : nullptr;
    if(!e1 || !e2) {
        return Point{};
    }

    const auto distp12 = ped2.position - ped1.position;
    const auto [distance, ep12] = distp12.NormAndNormalized();
    const double adjustedDist =
        distance - (ped1.radius.value_or(Defaults::radius) + ped2.radius.value_or(Defaults::radius));

    const auto& orientation1 = ped1.orientation.value_or(Point{0.0, 0.0});
    const auto& d1 = (routing1.destination - ped1.position).Normalized();
    const auto& orientation2 = ped2.orientation.value_or(Point{0.0, 0.0});
    const auto& velocity1 = ped1.velocity.value_or(Point{0.0, 0.0});
    const auto& velocity2 = ped2.velocity.value_or(Point{0.0, 0.0});

    const auto inPerceptionRange =
        d1.ScalarProduct(ep12) >= 0 || orientation1.ScalarProduct(ep12) >= 0;
    if(!inPerceptionRange) {
        return Point{};
    }

    const double S_Gap = (velocity1 - velocity2).ScalarProduct(ep12) * e1->anticipationTime;
    double R_dist = std::max(adjustedDist - S_Gap, 0.0);

    constexpr double alignmentBase = 1.0;
    constexpr double alignmentWeight = 0.5;
    const double alignmentFactor =
        alignmentBase + alignmentWeight * (1.0 - d1.ScalarProduct(orientation2));
    const auto strengthN =
        ped1.strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion);
    const auto rangeN = ped1.rangeNeighborRepulsion.value_or(Defaults::rangeNeighborRepulsion);
    const double interactionStrength = strengthN * alignmentFactor * std::exp(-R_dist / rangeN);
    const auto newep12 = distp12 + velocity2 * e2->anticipationTime;

    const auto influenceDirection = CalculateInfluenceDirection(d1, newep12);
    return influenceDirection * interactionStrength;
}

Point AnticipationVelocityModel::HandleWallAvoidance(
    const Point& direction,
    const Point& agentPosition,
    double agentRadius,
    const std::vector<LineSegment>& boundary,
    double wallBufferDistance,
    double pushoutStrength) const
{
    const double criticalWallDistance = wallBufferDistance + agentRadius;
    Point modifiedDirection = direction;

    for(const auto& wall : boundary) {
        const auto closestPoint = wall.ShortestPoint(agentPosition);
        const auto distanceVector = agentPosition - closestPoint;
        const auto [distance, normalTowardAgent] = distanceVector.NormAndNormalized();

        if(distance > criticalWallDistance) {
            continue;
        }
        const auto dotProduct = modifiedDirection.ScalarProduct(normalTowardAgent);
        if(dotProduct < 0) {
            const auto projectedDirection = modifiedDirection - normalTowardAgent * dotProduct;
            modifiedDirection = projectedDirection + normalTowardAgent * pushoutStrength;
        }
    }
    return modifiedDirection.Normalized();
}
