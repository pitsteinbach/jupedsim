// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "ParallelRuntime.hpp"
#include "RoutingEngine.hpp"
#include "Tracing.hpp"

#include <algorithm>
#include <execution>

#ifdef JPS_USE_OMP_PARALLEL_FOR
#include <omp.h>
#endif

class TacticalDecisionSystem
{
public:
    TacticalDecisionSystem() = default;
    ~TacticalDecisionSystem() = default;
    TacticalDecisionSystem(const TacticalDecisionSystem& other) = delete;
    TacticalDecisionSystem& operator=(const TacticalDecisionSystem& other) = delete;
    TacticalDecisionSystem(TacticalDecisionSystem&& other) = delete;
    TacticalDecisionSystem& operator=(TacticalDecisionSystem&& other) = delete;

    void Run(RoutingEngine& routingEngine, auto&& agents) const
    {
#ifdef JPS_USE_STD_PARALLEL_FOR
        std::for_each(
            std::execution::par,
            std::begin(agents),
            std::end(agents),
            [&routingEngine](auto& agent) {
                // compute and write back destination just like the non-std-parallel path
                agent.destination = routingEngine.ComputeWaypoint(agent.pos, agent.target);
            });
#elif defined(JPS_USE_CUSTOM_PARALLEL_FOR)
        ParallelRuntime::Instance().parallelFor(
            0, agents.size(), [&routingEngine, &agents](size_t idx) {
                const auto dest = agents[idx].target;
                agents[idx].destination = routingEngine.ComputeWaypoint(agents[idx].pos, dest);
            });
#elif defined(JPS_USE_OMP_PARALLEL_FOR)
#pragma omp parallel for
        for(auto& agent : agents) {
            // compute and write back destination just like the non-std-parallel path
            agent.destination = routingEngine.ComputeWaypoint(agent.pos, agent.target);
        }
#else
        for(auto& agent : agents) {
            // compute and write back destination just like the non-std-parallel path
            agent.destination = routingEngine.ComputeWaypoint(agent.pos, agent.target);
        }
#endif
    }
};
