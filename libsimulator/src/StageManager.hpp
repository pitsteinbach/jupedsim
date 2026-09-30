// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "GenericAgent.hpp"
#include "Geometry/Geometry.hpp"
#include "RoutingEngine.hpp"
#include "SimulationError.hpp"
#include "Stage.hpp"
#include "StageDescription.hpp"
#include "Visitor.hpp"

#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace detail
{
/// Put a stage's representative point on the surface that is closest to @p z_hint.
inline Location
locate_stage_point(const Geometry& geometry, Point point, std::string_view what, double z_hint)
{
    const auto location = geometry.get_location(point.x, point.y, z_hint);
    if(!location) {
        throw SimulationError("{} {} not inside walkable area", what, point);
    }
    return *location;
}

/// Locate @p point and register it with @p routing.
inline StageTarget point_target(
    const Geometry& geometry,
    RoutingEngine& routing,
    Point point,
    std::string_view what,
    double z_hint)
{
    const auto location = locate_stage_point(geometry, point, what, z_hint);
    return {routing.AddDestination(location), location};
}

inline std::vector<StageTarget> slot_targets(
    const Geometry& geometry,
    RoutingEngine& routing,
    const std::vector<Point>& slots,
    std::string_view what,
    double z_hint)
{
    std::vector<StageTarget> targets{};
    targets.reserve(slots.size());
    for(const auto& slot : slots) {
        targets.push_back(point_target(geometry, routing, slot, what, z_hint));
    }
    return targets;
}
} // namespace detail

class StageManager
{
private:
    std::unordered_map<BaseStage::ID, std::unique_ptr<BaseStage>> stages;

public:
    StageManager() {}
    ~StageManager() = default;
    StageManager(const StageManager& other) = delete;
    StageManager& operator=(const StageManager& other) = delete;
    StageManager(StageManager&& other) = delete;
    StageManager& operator=(StageManager&& other) = delete;

    BaseStage::ID AddStage(
        const StageDescription stageDescription,
        std::vector<GenericAgent::ID>& removedAgentsInLastIteration,
        const Geometry& geometry,
        RoutingEngine& routing,
        double z_hint)
    {
        std::unique_ptr<BaseStage> stage = std::visit(
            overloaded{
                [&](const WaypointDescription& d) -> std::unique_ptr<BaseStage> {
                    return std::make_unique<Waypoint>(
                        detail::point_target(geometry, routing, d.position, "WayPoint", z_hint),
                        d.distance);
                },
                [&](const ExitDescription& d) -> std::unique_ptr<BaseStage> {
                    auto pieces = geometry.split_into_region_pieces(d.polygon, z_hint);
                    if(pieces.empty()) {
                        throw SimulationError("Exit does not cover any walkable area.");
                    }
                    const StageTarget target{
                        routing.AddDestination(std::span<const AreaPiece>{pieces}),
                        geometry.anchor_of(pieces.front())};
                    return std::make_unique<Exit>(
                        std::move(pieces), target, removedAgentsInLastIteration);
                },
                [&](const NotifiableWaitingSetDescription& d) -> std::unique_ptr<BaseStage> {
                    return std::make_unique<NotifiableWaitingSet>(detail::slot_targets(
                        geometry, routing, d.slots, "NotifiableWaitingSet point", z_hint));
                },
                [&](const NotifiableQueueDescription& d) -> std::unique_ptr<BaseStage> {
                    return std::make_unique<NotifiableQueue>(detail::slot_targets(
                        geometry, routing, d.slots, "NotifiableQueue point", z_hint));
                },
                [](const DirectSteeringDescription&) -> std::unique_ptr<BaseStage> {
                    return std::make_unique<DirectSteering>();
                }},
            stageDescription);
        if(stages.find(stage->Id()) != stages.end()) {
            throw SimulationError("Internal error, stage id already in use.");
        }
        const auto id = stage->Id();
        stages.emplace(id, std::move(stage));

        return id;
    }

    void MigrateAgent(BaseStage::ID prevTarget, BaseStage::ID newTarget)
    {
        stages.at(newTarget)->IncreaseTargeting();
        stages.at(prevTarget)->DecreaseTargeting();
    }

    void HandleNewAgent(BaseStage::ID stageId) { stages.at(stageId)->IncreaseTargeting(); }
    void HandleRemoveAgent(BaseStage::ID stageId) { stages.at(stageId)->DecreaseTargeting(); }

    BaseStage* Stage(BaseStage::ID stageId) const
    {
        const auto iter = stages.find(stageId);
        if(iter == std::end(stages)) {
            throw SimulationError("Unknown stage id ({}) provided in journey.", stageId.getID());
        }
        return iter->second.get();
    }

    BaseStage* Stage(BaseStage::ID stageId)
    {
        auto iter = stages.find(stageId);
        if(iter == std::end(stages)) {
            throw SimulationError("Unknown stage id ({}) provided in journey.", stageId.getID());
        }
        return iter->second.get();
    }

    std::unordered_map<BaseStage::ID, std::unique_ptr<BaseStage>>& Stages() { return stages; }

    const std::unordered_map<BaseStage::ID, std::unique_ptr<BaseStage>>& Stages() const
    {
        return stages;
    }
};
