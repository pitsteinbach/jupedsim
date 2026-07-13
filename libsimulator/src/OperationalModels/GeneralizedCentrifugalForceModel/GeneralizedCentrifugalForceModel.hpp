// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "LineSegment.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"

class GeneralizedCentrifugalForceModel : public OperationalModel
{
public:
    struct Defaults {
        static constexpr double v0{1.2};
        static constexpr double mass{1.0};
        static constexpr double reactionTime{0.5}; // tau
        static constexpr double strengthNeighborRepulsion{0.3};
        static constexpr double strengthGeometryRepulsion{0.2};
        // GCFM-specific extras defaults
        static constexpr double Av{1.0};
        static constexpr double AMin{0.2};
        static constexpr double BMin{0.2};
        static constexpr double BMax{0.4};
        static constexpr double maxNeighborInteractionDistance{2};
        static constexpr double maxGeometryInteractionDistance{2};
        static constexpr double maxNeighborInterpolationDistance{0.1};
        static constexpr double maxGeometryInterpolationDistance{0.1};
        static constexpr double maxNeighborRepulsionForce{9};
        static constexpr double maxGeometryRepulsionForce{3};
    };

    static AgentState MakeState(Point pos = {});

private:
    double _cutOffRadius{4.0};

public:
    GeneralizedCentrifugalForceModel() = default;
    ~GeneralizedCentrifugalForceModel() override = default;

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
    Point ForceDriv(
        const AgentState& ped,
        Point target,
        double mass,
        double tau,
        double deltaT,
        Point& e0update) const;
    Point ForceRepPed(const AgentState& ped1, const GenericAgent& ped2) const;
    Point ForceRepRoom(const AgentState& ped, const CollisionGeometry& geometry) const;
    Point ForceRepWall(const AgentState& ped, const LineSegment& l) const;
    Point ForceRepStatPoint(const AgentState& ped, const Point& p, double l, double vn) const;
    Point ForceInterpolation(
        const AgentState& state,
        double v0,
        double K_ij,
        const Point& e,
        double v,
        double d,
        double r,
        double l) const;
    double AgentToAgentSpacing(const GenericAgent& agent, const GenericAgent& otherAgent) const;
    double AgentToAgentSpacingFromState(const AgentState& agent1, const GenericAgent& agent2) const;
};
