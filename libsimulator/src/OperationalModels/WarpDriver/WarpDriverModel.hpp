// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

class WarpDriverModel : public OperationalModel
{
public:
    struct Defaults {
        static constexpr double v0{1.2};
        static constexpr double radius{0.15};
        // Warp-specific extras defaults
        static constexpr double timeHorizon{2.0};
        static constexpr double stepSize{0.5};
        static constexpr double timeUncertainty{0.5};
        static constexpr double velocityUncertaintyX{0.2};
        static constexpr double velocityUncertaintyY{0.2};
        static constexpr int    numSamples{20};
    };

    static AgentState MakeState(Point pos = {});

    /// 3-component space-time point/vector used internally for collision probability field.
    struct SpaceTimePoint {
        double x{};
        double y{};
        double t{};
    };

private:
    struct IntrinsicField {
        std::vector<double> values;
        std::vector<Point> gradients;
        double xMin{-3.0};
        double xMax{3.0};
        double yMin{-3.0};
        double yMax{3.0};
        double dx{0.1};
        double dy{0.1};
        int nx{61};
        int ny{61};

        void Compute(double sigma);
        std::pair<double, Point> Sample(double x, double y) const;
    };

    double _cutOffRadius;
    IntrinsicField _intrinsicField;
    mutable std::mt19937 _rng;

public:
    WarpDriverModel(double sigma, uint64_t rngSeed = 42);
    ~WarpDriverModel() override = default;

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
};
