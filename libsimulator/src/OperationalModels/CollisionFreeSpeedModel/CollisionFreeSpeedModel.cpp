// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CollisionFreeSpeedModel.hpp"

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

AgentState CollisionFreeSpeedModel::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::COLLISION_FREE_SPEED,
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

OperationalModelType CollisionFreeSpeedModel::Type() const
{
    return OperationalModelType::COLLISION_FREE_SPEED;
}

void CollisionFreeSpeedModel::ComputeNext(
    double dT,
    const GenericAgent& current,
    GenericAgent& next,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    auto neighborhood = neighborhoodSearch.GetNeighboringAgents(Pos(current), _cutOffRadius);
    const auto& boundary = geometry.LineSegmentsInApproxDistanceTo(Pos(current));

    neighborhood.erase(
        std::remove_if(
            std::begin(neighborhood),
            std::end(neighborhood),
            [&current, &boundary](const auto& neighbor) {
                if(current.id == neighbor.id) {
                    return true;
                }
                const auto agent_to_neighbor = LineSegment(Pos(current), Pos(neighbor));
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
            return res + NeighborRepulsion(current, neighbor);
        });

    const auto boundaryRepulsion = std::accumulate(
        boundary.cbegin(),
        boundary.cend(),
        Point(0, 0),
        [this, &current](const auto& acc, const auto& element) {
            return acc + BoundaryRepulsion(current, element);
        });

    const auto& state = std::get<AgentState>(current.model);
    const auto desired_direction = (current.destination - Pos(current)).Normalized();
    auto direction = (desired_direction + neighborRepulsion + boundaryRepulsion).Normalized();
    if(direction == Point{}) {
        direction = state.orientation.value_or(Point{0.0, 0.0});
    }
    const auto spacing = std::accumulate(
        std::begin(neighborhood),
        std::end(neighborhood),
        std::numeric_limits<double>::max(),
        [&current, &direction, this](const auto& res, const auto& neighbor) {
            return std::min(res, GetSpacing(current, neighbor, direction));
        });

    const auto optimal_speed =
        OptimalSpeed(current, spacing, state.timeGap.value_or(Defaults::timeGap));
    const auto velocity = direction * optimal_speed;
    auto& nextState = std::get<AgentState>(next.model);
    nextState.position = Pos(current) + velocity * dT;
    nextState.orientation = direction;
}

void CollisionFreeSpeedModel::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    const auto& state = std::get<AgentState>(agent.model);
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

double CollisionFreeSpeedModel::OptimalSpeed(
    const GenericAgent& ped,
    double spacing,
    double time_gap) const
{
    const auto& state = std::get<AgentState>(ped.model);
    return std::min(std::max(spacing / time_gap, 0.0), state.v0.value_or(Defaults::v0));
}

double CollisionFreeSpeedModel::GetSpacing(
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
    if(direction.ScalarProduct(distp12) < 0) {
        return std::numeric_limits<double>::max();
    }
    const auto left = direction.Rotate90Deg();
    const auto l = s1->radius.value_or(Defaults::radius) + s2->radius.value_or(Defaults::radius);
    if(std::abs(left.ScalarProduct(distp12)) > l) {
        return std::numeric_limits<double>::max();
    }
    return distp12.Norm() - l;
}

Point CollisionFreeSpeedModel::NeighborRepulsion(
    const GenericAgent& ped1,
    const GenericAgent& ped2) const
{
    const auto* s1 = std::get_if<AgentState>(&ped1.model);
    const auto* s2 = std::get_if<AgentState>(&ped2.model);
    if(!s1 || !s2) {
        return Point{};
    }
    const auto distp12 = Pos(ped2) - Pos(ped1);
    const auto [distance, direction] = distp12.NormAndNormalized();
    const auto l = s1->radius.value_or(Defaults::radius) + s2->radius.value_or(Defaults::radius);
    const auto strengthN = s1->strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion);
    const auto rangeN = s1->rangeNeighborRepulsion.value_or(Defaults::rangeNeighborRepulsion);
    return direction * -(strengthN * std::exp((l - distance) / rangeN));
}

Point CollisionFreeSpeedModel::BoundaryRepulsion(
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
