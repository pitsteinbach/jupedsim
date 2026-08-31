// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef JUPEDSIM_PARALLEL_STL
#include <execution>
#endif

#include "AgentView.hpp"
#include "EnvironmentQuery.hpp"
#include "GenericAgent.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

class OperationalDecisionSystem
{
    std::unique_ptr<OperationalModel> _model{};
    AgentContainer<GenericAgent> _next{};
    std::vector<size_t> _workIndices{};

public:
    OperationalDecisionSystem(std::unique_ptr<OperationalModel>&& model) : _model(std::move(model))
    {
    }
    ~OperationalDecisionSystem() = default;
    OperationalDecisionSystem(const OperationalDecisionSystem& other) = delete;
    OperationalDecisionSystem& operator=(const OperationalDecisionSystem& other) = delete;
    OperationalDecisionSystem(OperationalDecisionSystem&& other) = delete;
    OperationalDecisionSystem& operator=(OperationalDecisionSystem&& other) = delete;

    OperationalModelType ModelType() const { return _model->Type(); }

    void
    Run(double dT,
        double /*t_in_sec*/,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
        const CollisionGeometry& geometry,
        AgentContainer<GenericAgent>& agents)
    {
        const EnvironmentQuery envQuery{geometry, neighborhoodSearch};
        _next.clear();
        std::copy(std::begin(agents), std::end(agents), std::back_inserter(_next));

        _workIndices.resize(agents.size());
        std::iota(_workIndices.begin(), _workIndices.end(), size_t{0});

        // Each index is owned by exactly one invocation: reads agents[i] (immutable snapshot),
        // writes _next[i] (disjoint). envQuery and _model are read-only.
        const auto processAgent = [&](size_t index) {
            const auto& current = agents[index];
            auto& next = _next[index];
            const AgentStep step{envQuery, current, dT};
            const Point movement = _model->ComputeNextState(current.state, next.state, step);
            next.MoveAlongSurface(movement);
        };

#ifdef JUPEDSIM_PARALLEL_STL
        std::for_each(std::execution::par, _workIndices.begin(), _workIndices.end(), processAgent);
#else
        std::for_each(_workIndices.begin(), _workIndices.end(), processAgent);
#endif
        // Swap in the computed generation. This is safe because no caller retains
        // pointers/references across an iteration (Python-side agent handles resolve per
        // access) and Simulation::Iterate rebuilds the neighborhood grid right after this
        // step.
        agents.swap(_next);
    }

    void ValidateAgent(
        const GenericAgent& agent,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
        const CollisionGeometry& geometry) const
    {
        const EnvironmentQuery envQuery{geometry, neighborhoodSearch};
        _model->CheckModelConstraint(agent, AgentView{envQuery, agent});
    }
};
