// SPDX-License-Identifier: LGPL-3.0-or-later
#include "SocialForceModel.hpp"

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "LineSegment.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"
#include "SimulationError.hpp"

#include <cmath>
#include <iterator>
#include <numeric>
#include <string>

AgentState SocialForceModel::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::SOCIAL_FORCE,
        .position = pos,
        .v0 = Defaults::v0,
        .radius = Defaults::radius,
        .velocity = Point{0.0, 0.0},
        .mass = Defaults::mass,
        .reactionTime = Defaults::reactionTime,
        .extras = SFMExtras{
            .agentScale = Defaults::agentScale,
            .obstacleScale = Defaults::obstacleScale,
            .forceDistance = Defaults::forceDistance,
            .bodyForce = Defaults::bodyForce,
            .friction = Defaults::friction,
        },
    };
}

OperationalModelType SocialForceModel::Type() const
{
    return OperationalModelType::SOCIAL_FORCE;
}

void SocialForceModel::ComputeNext(
    double dT,
    const GenericAgent& current,
    GenericAgent& next,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    const auto& state = current.model;
    const auto mass = state.mass.value_or(Defaults::mass);
    auto forces = DrivingForce(current);

    const auto neighborhood =
        neighborhoodSearch.GetNeighboringAgents(Pos(current), this->_cutOffRadius);
    Point F_rep;
    for(const auto& neighbor : neighborhood) {
        if(neighbor.id == current.id) {
            continue;
        }
        F_rep += AgentForce(current, neighbor);
    }
    forces += F_rep / mass;
    const auto& walls = geometry.LineSegmentsInApproxDistanceTo(Pos(current));

    const auto obstacle_f = std::accumulate(
        walls.cbegin(),
        walls.cend(),
        Point(0, 0),
        [this, &current](const auto& acc, const auto& element) {
            return acc + ObstacleForce(current, element);
        });
    forces += obstacle_f / mass;

    const auto currentVelocity = state.velocity.value_or(Point{0.0, 0.0});
    const auto velocity = currentVelocity + forces * dT;
    auto& nextState = next.model;
    nextState.position = Pos(current) + velocity * dT;
    nextState.velocity = velocity;
}

void SocialForceModel::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    auto throwIfNegative = [](double value, std::string name) {
        if(value < 0) {
            throw SimulationError(
                "Model constraint violation: {} {} not in allowed range, "
                "{} needs to be positive",
                name,
                value,
                name);
        }
    };

    const auto& state = agent.model;
    throwIfNegative(state.mass.value_or(Defaults::mass), "mass");
    throwIfNegative(state.v0.value_or(Defaults::v0), "desired speed");
    throwIfNegative(state.reactionTime.value_or(Defaults::reactionTime), "reaction time");
    const auto radius = state.radius.value_or(Defaults::radius);
    throwIfNegative(radius, "radius");

    const auto neighbors = neighborhoodSearch.GetNeighboringAgents(Pos(agent), 2);
    for(const auto& neighbor : neighbors) {
        const auto distance = (Pos(agent) - Pos(neighbor)).Norm();
        if(radius >= distance) {
            throw SimulationError(
                "Model constraint violation: Agent {} too close to agent {}: distance {}, "
                "radius {}",
                Pos(agent),
                Pos(neighbor),
                distance,
                radius);
        }
    }
    const auto maxRadius = radius / 2;
    const auto lineSegments = geometry.LineSegmentsInDistanceTo(maxRadius, Pos(agent));
    if(std::begin(lineSegments) != std::end(lineSegments)) {
        throw SimulationError(
            "Model constraint violation: Agent {} too close to geometry boundaries, distance <= "
            "{}/2",
            Pos(agent),
            radius);
    }
}

Point SocialForceModel::DrivingForce(const GenericAgent& agent)
{
    const auto& state = agent.model;
    const Point e0 = (agent.destination - Pos(agent)).Normalized();
    const auto v0 = state.v0.value_or(Defaults::v0);
    const auto reactionTime = state.reactionTime.value_or(Defaults::reactionTime);
    const auto velocity = state.velocity.value_or(Point{0.0, 0.0});
    return (e0 * v0 - velocity) / reactionTime;
}

double SocialForceModel::PushingForceLength(double A, double B, double r, double distance)
{
    return A * std::exp((r - distance) / B);
}

Point SocialForceModel::AgentForce(const GenericAgent& ped1, const GenericAgent& ped2) const
{
    const auto& s1 = ped1.model;
    const auto& s2 = ped2.model;
    const auto& e1 = std::get<SFMExtras>(*s1.extras);

    const double total_radius =
        s1.radius.value_or(Defaults::radius) + s2.radius.value_or(Defaults::radius);
    const auto v1 = s1.velocity.value_or(Point{0.0, 0.0});
    const auto v2 = s2.velocity.value_or(Point{0.0, 0.0});

    return ForceBetweenPoints(
        Pos(ped1),
        Pos(ped2),
        e1.agentScale,
        e1.forceDistance,
        total_radius,
        v2 - v1,
        e1.bodyForce,
        e1.friction);
}

Point SocialForceModel::ObstacleForce(const GenericAgent& agent, const LineSegment& segment) const
{
    const auto& state = agent.model;
    const auto& extras = std::get<SFMExtras>(*state.extras);
    const Point pt = segment.ShortestPoint(Pos(agent));
    return ForceBetweenPoints(
        Pos(agent),
        pt,
        extras.obstacleScale,
        extras.forceDistance,
        state.radius.value_or(Defaults::radius),
        state.velocity.value_or(Point{0.0, 0.0}),
        extras.bodyForce,
        extras.friction);
}

Point SocialForceModel::ForceBetweenPoints(
    const Point pt1,
    const Point pt2,
    const double A,
    const double B,
    const double radius,
    const Point velocity,
    const double bodyForce,
    const double friction)
{
    const double dist = (pt1 - pt2).Norm();
    double pushing_force_length = PushingForceLength(A, B, radius, dist);
    double friction_force_length = 0;
    const Point n_ij = (pt1 - pt2).Normalized();
    const Point tangent = n_ij.Rotate90Deg();
    if(dist < radius) {
        pushing_force_length += bodyForce * (radius - dist);
        friction_force_length = friction * (radius - dist) * (velocity.ScalarProduct(tangent));
    }
    return n_ij * pushing_force_length + tangent * friction_force_length;
}
