// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "LineSegment.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"

class SocialForceModel : public OperationalModel
{
public:
    struct Defaults {
        static constexpr double v0{0.8};         // desiredSpeed
        static constexpr double mass{80.0};
        static constexpr double reactionTime{0.5}; // tau
        static constexpr double radius{0.3};
        // SFM-specific extras defaults
        static constexpr double agentScale{2000.0};
        static constexpr double obstacleScale{2000.0};
        static constexpr double forceDistance{0.08};
        static constexpr double bodyForce{120000};
        static constexpr double friction{240000};
    };

    static AgentState MakeState(Point pos = {});

private:
    double _cutOffRadius{2.5};

public:
    SocialForceModel() = default;
    ~SocialForceModel() override = default;
    OperationalModelType Type() const override;
    void ComputeNext(
        double dT,
        const AgentState& current,
        AgentState& next,
        const AgentRouting& routing,
        const CollisionGeometry& geometry,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const override;
    void CheckModelConstraint(
        const GenericAgent& agent,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
        const CollisionGeometry& geometry) const override;

private:
    static Point DrivingForce(const AgentState& agent, const Point& destination);
    Point AgentForce(const AgentState& ped1, const GenericAgent& ped2) const;
    Point ObstacleForce(const AgentState& agent, const LineSegment& segment) const;
    static Point ForceBetweenPoints(
        const Point pt1,
        const Point pt2,
        const double A,
        const double B,
        const double radius,
        const Point velocity,
        const double bodyForce,
        const double friction);
    static double PushingForceLength(double A, double B, double r, double distance);
};
