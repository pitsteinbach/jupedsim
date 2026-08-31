// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "Routing.hpp"
#ifdef JUPEDSIM_PARALLEL_STL
#include <execution>
#endif
#include <algorithm>

class TacticalDecisionSystem
{
public:
    TacticalDecisionSystem() = default;
    ~TacticalDecisionSystem() = default;
    TacticalDecisionSystem(const TacticalDecisionSystem& other) = delete;
    TacticalDecisionSystem& operator=(const TacticalDecisionSystem& other) = delete;
    TacticalDecisionSystem(TacticalDecisionSystem&& other) = delete;
    TacticalDecisionSystem& operator=(TacticalDecisionSystem&& other) = delete;

    void Run(Router& router, auto&& agents) const
    {

        // #ifdef JUPEDSIM_PARALLEL_STL
        // std::for_each(std::execution::par, agents.begin(), agents.end(), [&router](auto& agent) {
        // if(agent.destinationId == Router::DirectSteeringId) {
        // agent.nextTarget = router.ComputeWaypoint(agent.Position(), agent.finalTarget);
        //} else {
        // agent.nextTarget = router.ComputeWaypoint(agent.Position(), agent.destinationId);
        //}
        //});
        // #else
        for(auto& agent : agents) {
            if(agent.destinationId == Router::DirectSteeringId) {
                agent.nextTarget = router.ComputeWaypoint(agent.Position(), agent.finalTarget);
            } else {
                agent.nextTarget = router.ComputeWaypoint(agent.Position(), agent.destinationId);
            }
        }
        // #endif
    }
};
