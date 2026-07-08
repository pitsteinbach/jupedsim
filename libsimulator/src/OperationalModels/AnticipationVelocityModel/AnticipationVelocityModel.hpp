// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "LineSegment.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"

#include <cstdint>
#include <random>
#include <vector>

class AnticipationVelocityModel : public OperationalModel
{
public:
    struct Defaults {
        static constexpr double v0{1.2};
        static constexpr double radius{0.2};
        static constexpr double timeGap{1.06};
        static constexpr double strengthNeighborRepulsion{8.0};
        static constexpr double rangeNeighborRepulsion{0.1};
        static constexpr double reactionTime{0.3};
        // AVM-specific
        static constexpr double wallBufferDistance{0.1};
        static constexpr double anticipationTime{1.0};
        static constexpr double pushoutStrength{0.3};
    };

    static AgentState MakeState(Point pos = {});

private:
    double _cutOffRadius{3};
    mutable std::mt19937 gen;

public:
    explicit AnticipationVelocityModel(uint64_t rng_seed);
    ~AnticipationVelocityModel() override = default;
    OperationalModelType Type() const override;
    void ComputeNext(
        double dT,
        const GenericAgent& current,
        GenericAgent& next,
        const CollisionGeometry& geometry,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const override;
    void CheckModelConstraint(
        const GenericAgent& agent,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
        const CollisionGeometry& geometry) const override;

private:
    double OptimalSpeed(const GenericAgent& ped, double spacing, double time_gap) const;
    Point CalculateInfluenceDirection(
        const Point& desiredDirection,
        const Point& predictedDirection) const;
    double
    GetSpacing(const GenericAgent& ped1, const GenericAgent& ped2, const Point& direction) const;
    Point NeighborRepulsion(const GenericAgent& ped1, const GenericAgent& ped2) const;

    Point HandleWallAvoidance(
        const Point& direction,
        const Point& agentPosition,
        double agentRadius,
        const std::vector<LineSegment>& boundary,
        double wallBufferDistance,
        double pushoutStrength) const;

    Point
    UpdateDirection(const GenericAgent& ped, const Point& calculatedDirection, double dt) const;
};
