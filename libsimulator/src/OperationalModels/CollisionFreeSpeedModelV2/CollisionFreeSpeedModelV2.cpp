// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CollisionFreeSpeedModelV2.hpp"

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

AgentState CollisionFreeSpeedModelV2::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::COLLISION_FREE_SPEED_V2,
        .position = pos,
        .orientation = Point{0.0, 0.0},
        .v0 = Defaults::v0,
        .radius = Defaults::radius,
        .timeGap = Defaults::timeGap,
        .strengthNeighborRepulsion = Defaults::strengthNeighborRepulsion,
        .rangeNeighborRepulsion = Defaults::rangeNeighborRepulsion,
        .strengthGeometryRepulsion = Defaults::strengthGeometryRepulsion,
        .rangeGeometryRepulsion = Defaults::rangeGeometryRepulsion,
    };
}

OperationalModelType CollisionFreeSpeedModelV2::Type() const
{
    return OperationalModelType::COLLISION_FREE_SPEED_V2;
}

void CollisionFreeSpeedModelV2::ComputeNext(
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
        [&current, this](const auto& res, const auto& neighbor) {
            return res + NeighborRepulsion(current, neighbor.model);
        });

    const auto boundaryRepulsion = std::accumulate(
        boundary.cbegin(),
        boundary.cend(),
        Point(0, 0),
        [this, &current](const auto& acc, const auto& element) {
            return acc + BoundaryRepulsion(current, element);
        });

    const auto desired_direction = (routing.destination - current.position).Normalized();
    auto direction = (desired_direction + neighborRepulsion + boundaryRepulsion).Normalized();
    if(direction == Point{}) {
        direction = current.orientation.value_or(Point{0.0, 0.0});
    }
    const auto spacing = std::accumulate(
        std::begin(neighborhood),
        std::end(neighborhood),
        std::numeric_limits<double>::max(),
        [&current, &direction, this](const auto& res, const auto& neighbor) {
            return std::min(res, GetSpacing(current, neighbor.model, direction));
        });

    const auto optimal_speed =
        OptimalSpeed(current, spacing, current.timeGap.value_or(Defaults::timeGap));
    const auto velocity = direction * optimal_speed;
    next.position = current.position + velocity * dT;
    next.orientation = direction;
}

void CollisionFreeSpeedModelV2::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    const auto& state = agent.model;
    const auto r = state.radius.value_or(Defaults::radius);
    constexpr double rMin = 0.;
    constexpr double rMax = 2.;
    validateConstraint(r, rMin, rMax, "radius", true);

    const auto v0 = state.v0.value_or(Defaults::v0);
    constexpr double v0Min = 0.;
    constexpr double v0Max = 10.;
    validateConstraint(v0, v0Min, v0Max, "v0");

    const auto timeGap = state.timeGap.value_or(Defaults::timeGap);
    constexpr double timeGapMin = 0.1;
    constexpr double timeGapMax = 10.;
    validateConstraint(timeGap, timeGapMin, timeGapMax, "timeGap");

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

double CollisionFreeSpeedModelV2::OptimalSpeed(
    const AgentState& ped,
    double spacing,
    double time_gap) const
{
    return std::min(std::max(spacing / time_gap, 0.0), ped.v0.value_or(Defaults::v0));
}

double CollisionFreeSpeedModelV2::GetSpacing(
    const AgentState& ped1,
    const AgentState& ped2,
    const Point& direction) const
{
    const auto distp12 = ped2.position - ped1.position;
    if(direction.ScalarProduct(distp12) < 0) {
        return std::numeric_limits<double>::max();
    }
    const auto left = direction.Rotate90Deg();
    const auto l = ped1.radius.value_or(Defaults::radius) + ped2.radius.value_or(Defaults::radius);
    if(std::abs(left.ScalarProduct(distp12)) > l) {
        return std::numeric_limits<double>::max();
    }
    return distp12.Norm() - l;
}

Point CollisionFreeSpeedModelV2::NeighborRepulsion(
    const AgentState& ped1,
    const AgentState& ped2) const
{
    const auto distp12 = ped2.position - ped1.position;
    const auto [distance, direction] = distp12.NormAndNormalized();
    const auto l = ped1.radius.value_or(Defaults::radius) + ped2.radius.value_or(Defaults::radius);
    const auto strengthN = ped1.strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion);
    const auto rangeN = ped1.rangeNeighborRepulsion.value_or(Defaults::rangeNeighborRepulsion);
    return direction * -(strengthN * std::exp((l - distance) / rangeN));
}

Point CollisionFreeSpeedModelV2::BoundaryRepulsion(
    const AgentState& ped,
    const LineSegment& boundary_segment) const
{
    const auto pt = boundary_segment.ShortestPoint(ped.position);
    const auto dist_vec = pt - ped.position;
    const auto [dist, e_iw] = dist_vec.NormAndNormalized();
    const auto l = ped.radius.value_or(Defaults::radius);
    const auto strengthG = ped.strengthGeometryRepulsion.value_or(Defaults::strengthGeometryRepulsion);
    const auto rangeG = ped.rangeGeometryRepulsion.value_or(Defaults::rangeGeometryRepulsion);
    return e_iw * (-strengthG * std::exp((l - dist) / rangeG));
}
