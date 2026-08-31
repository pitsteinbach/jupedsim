// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "Journey.hpp"
#include "Point.hpp"
#include "Routing.hpp"
#include "StageManager.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// Unique set of routing destinations collected during the strategical phase.
/// Passed to Router::PrecomputeDestinations so floor fields can be computed in
/// parallel before the per-agent routing loop runs.
struct AgentDestinations {
    std::vector<size_t> ids; // unique polygon destination IDs (non-DirectSteering)
    std::vector<Point> points; // point targets for DirectSteering agents
};

class StrategicalDecisionSystem
{
public:
    StrategicalDecisionSystem() = default;
    ~StrategicalDecisionSystem() = default;
    StrategicalDecisionSystem(const StrategicalDecisionSystem& other) = delete;
    StrategicalDecisionSystem& operator=(const StrategicalDecisionSystem& other) = delete;
    StrategicalDecisionSystem(StrategicalDecisionSystem&& other) = delete;
    StrategicalDecisionSystem& operator=(StrategicalDecisionSystem&& other) = delete;

    AgentDestinations
    Run(const std::unordered_map<Journey::ID, std::unique_ptr<Journey>>& journeys,
        auto&& agents,
        StageManager& stageManager) const
    {
        AgentDestinations dests;
        std::unordered_set<size_t> seenIds;
        for(auto& agent : agents) {
            const auto [target, destId, id] = journeys.at(agent.journeyId)->Target(agent);
            agent.finalTarget = target;
            agent.destinationId = destId;
            stageManager.MigrateAgent(agent.stageId, id);
            agent.stageId = id;
            if(destId != Router::DirectSteeringId) {
                if(seenIds.insert(destId).second) {
                    dests.ids.push_back(destId);
                }
            } else {
                dests.points.push_back(target);
            }
        }
        return dests;
    }
};
