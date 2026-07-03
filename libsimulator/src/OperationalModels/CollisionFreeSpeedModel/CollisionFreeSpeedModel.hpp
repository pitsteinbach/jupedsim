// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "CollisionGeometry.hpp"
#include "LineSegment.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"

#include <fmt/core.h>

class CollisionFreeSpeedModel : public OperationalModel
{
public:
    /// Per-agent state of the collision free speed model.
    struct Agent {
        Point orientation{0.0, 0.0};
        double timeGap{1};
        double v0{1.2};
        double radius{0.2};
        // Configured simulation-wide via the builder; stamped into every agent
        // by InitializeAgent, not settable per agent through the Python API.
        double strengthNeighborRepulsion{8.0};
        double rangeNeighborRepulsion{0.1};
        double strengthGeometryRepulsion{5.0};
        double rangeGeometryRepulsion{0.02};
    };

private:
    double _cutOffRadius{3};
    double strengthNeighborRepulsion;
    double rangeNeighborRepulsion;
    double strengthGeometryRepulsion;
    double rangeGeometryRepulsion;

public:
    CollisionFreeSpeedModel(
        double strengthNeighborRepulsion,
        double rangeNeighborRepulsion,
        double strengthGeometryRepulsion,
        double rangeGeometryRepulsion);
    ~CollisionFreeSpeedModel() override = default;
    OperationalModelType Type() const override;
    void InitializeAgent(GenericAgent& agent) const override;
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
    double
    GetSpacing(const GenericAgent& ped1, const GenericAgent& ped2, const Point& direction) const;
    Point NeighborRepulsion(const GenericAgent& ped1, const GenericAgent& ped2) const;
    Point BoundaryRepulsion(const GenericAgent& ped, const LineSegment& boundary_segment) const;
};

template <>
struct fmt::formatter<CollisionFreeSpeedModel::Agent> {

    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const CollisionFreeSpeedModel::Agent& m, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "CollisionFreeSpeedModel[orientation={}, timeGap={}, v0={}, radius={}])",
            m.orientation,
            m.timeGap,
            m.v0,
            m.radius);
    }
};
