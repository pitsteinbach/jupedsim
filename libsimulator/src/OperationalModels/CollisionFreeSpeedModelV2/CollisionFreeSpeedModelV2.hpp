// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "LineSegment.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"

class CollisionFreeSpeedModelV2 : public OperationalModel
{
public:
    struct Defaults {
        static constexpr double v0{1.2};
        static constexpr double radius{0.2};
        static constexpr double timeGap{1.0};
        static constexpr double strengthNeighborRepulsion{8.0};
        static constexpr double rangeNeighborRepulsion{0.1};
        static constexpr double strengthGeometryRepulsion{5.0};
        static constexpr double rangeGeometryRepulsion{0.02};
    };

    static AgentState MakeState(Point pos = {});

private:
    double _cutOffRadius{3};

public:
    CollisionFreeSpeedModelV2() = default;
    ~CollisionFreeSpeedModelV2() override = default;
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
    double OptimalSpeed(const AgentState& ped, double spacing, double time_gap) const;
    double
    GetSpacing(const AgentState& ped1, const AgentState& ped2, const Point& direction) const;
    Point NeighborRepulsion(const AgentState& ped1, const AgentState& ped2) const;
    Point BoundaryRepulsion(const AgentState& ped, const LineSegment& boundary_segment) const;
};
