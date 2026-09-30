// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Stage.hpp"

#include "GenericAgent.hpp"
#include "Point.hpp"
#include "Polygon.hpp"
#include "Simulation.hpp"
#include "SimulationError.hpp"
#include "Util.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <list>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
/// Base Proxy
////////////////////////////////////////////////////////////////////////////////
size_t BaseProxy::CountTargeting() const
{
    return stage->CountTargeting();
}

////////////////////////////////////////////////////////////////////////////////
/// NotifiableQueueProxy
////////////////////////////////////////////////////////////////////////////////

size_t NotifiableQueueProxy::CountEnqueued() const
{
    auto concreteStage = dynamic_cast<NotifiableQueue*>(stage);
    assert(stage);
    return concreteStage->Occupants().size();
}

const std::vector<GenericAgent::ID>& NotifiableQueueProxy::Enqueued() const
{
    const auto concreteStage = dynamic_cast<const NotifiableQueue*>(stage);
    assert(stage);
    return concreteStage->Occupants();
}

void NotifiableQueueProxy::Pop(size_t count)
{
    auto concreteStage = dynamic_cast<NotifiableQueue*>(stage);
    assert(stage);
    return concreteStage->Pop(count);
}

////////////////////////////////////////////////////////////////////////////////
/// NotifiableWaitingSetProxy
////////////////////////////////////////////////////////////////////////////////
void NotifiableWaitingSetProxy::State(WaitingSetState newState)
{
    auto concreteStage = dynamic_cast<NotifiableWaitingSet*>(stage);
    assert(stage);
    concreteStage->State(newState);
}

WaitingSetState NotifiableWaitingSetProxy::State() const
{
    const auto concreteStage = dynamic_cast<const NotifiableWaitingSet*>(stage);
    assert(stage);
    return concreteStage->State();
}

size_t NotifiableWaitingSetProxy::CountWaiting() const
{
    const auto concreteStage = dynamic_cast<const NotifiableWaitingSet*>(stage);
    assert(stage);
    return concreteStage->Occupants().size();
}

const std::vector<GenericAgent::ID>& NotifiableWaitingSetProxy::Waiting() const
{
    auto concreteStage = dynamic_cast<NotifiableWaitingSet*>(stage);
    assert(stage);
    return concreteStage->Occupants();
}

////////////////////////////////////////////////////////////////////////////////
/// Waypoint
////////////////////////////////////////////////////////////////////////////////
Waypoint::Waypoint(StageTarget target_, double distance_) : target(target_), distance(distance_)
{
}

bool Waypoint::IsCompleted(const GenericAgent& agent)
{
    return agent.location.distance_to(target.anchor) <= distance;
}

StageTarget Waypoint::Target(const GenericAgent&)
{
    return target;
}

StageProxy Waypoint::Proxy(Simulation* simulation)
{
    return WaypointProxy(simulation, this);
}

////////////////////////////////////////////////////////////////////////////////
/// Exit
////////////////////////////////////////////////////////////////////////////////
Exit::Exit(
    std::vector<AreaPiece> areas_,
    StageTarget target_,
    std::vector<GenericAgent::ID>& toRemove_)
    : areas(std::move(areas_)), target(target_), toRemove(toRemove_)
{
    if(areas.empty()) {
        throw SimulationError("Exit area does not cover any walkable area.");
    }
}

namespace
{
/// Inside or on the boundary of @p area, holes excluded.
bool covers(const PolyWithHoles& area, Point p)
{
    const Point2D q{p.x, p.y};
    if(area.outer_boundary().bounded_side(q) == CGAL::ON_UNBOUNDED_SIDE) {
        return false;
    }
    return std::none_of(area.holes_begin(), area.holes_end(), [&q](const Poly& hole) {
        return hole.bounded_side(q) == CGAL::ON_BOUNDED_SIDE;
    });
}
} // namespace

bool Exit::IsCompleted(const GenericAgent& agent)
{
    // Pieces lie in one region each, and a region never overlaps itself in plan: the region
    // check is what tells the exit's floor from the ones above and below it.
    const bool hasReachedExit =
        std::any_of(areas.begin(), areas.end(), [&agent](const AreaPiece& piece) {
            return piece.region == agent.location.region() &&
                   covers(piece.polygon, agent.location.xy());
        });
    if(hasReachedExit) {
        toRemove.push_back(agent.id);
    }
    return hasReachedExit;
}

StageTarget Exit::Target(const GenericAgent&)
{
    return target;
}

StageProxy Exit::Proxy(Simulation* simulation)
{
    return ExitProxy(simulation, this);
}

////////////////////////////////////////////////////////////////////////////////
/// NotifiableWaitingSet
////////////////////////////////////////////////////////////////////////////////
NotifiableWaitingSet::NotifiableWaitingSet(std::vector<StageTarget> slots_)
    : slots(std::move(slots_))
{
    occupants.reserve(slots.size());
}

bool NotifiableWaitingSet::IsCompleted(const GenericAgent& agent)
{
    if(state == WaitingSetState::Active) {
        return false;
    }
    const auto find_iter = std::find(std::begin(occupants), std::end(occupants), agent.id);
    if(find_iter != std::end(occupants)) {
        return true;
    }
    return agent.location.distance_to(slots[0].anchor) <= 1;
}

StageTarget NotifiableWaitingSet::Target(const GenericAgent& agent)
{
    if(state == WaitingSetState::Inactive) {
        return slots[0];
    }

    const auto next_slot_index = std::min(occupants.size(), slots.size() - 1);

    for(size_t index = 0; index < next_slot_index; ++index) {
        if(agent.id == occupants[index]) {
            return slots[index];
        }
    }

    return slots[next_slot_index];
}

void NotifiableWaitingSet::State(WaitingSetState s)
{
    if(state == s) {
        return;
    }
    if(s == WaitingSetState::Active) {
        occupants.clear();
    }
    state = s;
}

WaitingSetState NotifiableWaitingSet::State() const
{
    return state;
}

StageProxy NotifiableWaitingSet::Proxy(Simulation* simulation)
{
    return NotifiableWaitingSetProxy(simulation, this);
}

const std::vector<GenericAgent::ID>& NotifiableWaitingSet::Occupants() const
{
    return occupants;
}

void NotifiableWaitingSet::Update(const EnvironmentQuery& envQuery)
{
    if(state == WaitingSetState::Inactive) {
        return;
    }
    const auto count_occupants = occupants.size();
    if(count_occupants == slots.size()) {
        return;
    }

    for(size_t index = count_occupants; index < slots.size(); ++index) {
        const auto& slot = slots[index].anchor;
        auto candidates = envQuery.AgentsInRange(slot.xy(), 2, [&](const GenericAgent& candidate) {
            return envQuery.NoGeometryBetween(slot, candidate.location);
        });

        GenericAgent::ID occupant = GenericAgent::ID::Invalid;
        double min_distance = std::numeric_limits<double>::max();
        for(const auto& agent : candidates) {
            if(agent.stageId == id) {
                if(std::find(std::begin(occupants), std::end(occupants), agent.id) ==
                   std::end(occupants)) {
                    const auto distance = (agent.location.xy() - slot.xy()).Norm();
                    if(distance < min_distance) {
                        min_distance = distance;
                        occupant = agent.id;
                    }
                }
            }
        }
        if(occupant != GenericAgent::ID::Invalid) {
            occupants.push_back(occupant);
        } else {
            return;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// NotifiablQueue
////////////////////////////////////////////////////////////////////////////////
NotifiableQueue::NotifiableQueue(std::vector<StageTarget> slots_) : slots(std::move(slots_))
{
}

bool NotifiableQueue::IsCompleted(const GenericAgent& agent)
{
    const bool completed = exitingThisUpdate.contains(agent.id);
    if(completed) {
        exitingThisUpdate.erase(agent.id);
    }
    return completed;
}

StageTarget NotifiableQueue::Target(const GenericAgent& agent)
{

    if(const auto index_opt = IndexInContainer(occupants, agent.id); index_opt) {
        return slots[*index_opt];
    }

    const auto next_target_index = std::min(occupants.size(), slots.size() - 1);
    return slots[next_target_index];
}

void NotifiableQueue::Pop(size_t count)
{
    for(size_t counter = 0; counter < count; ++counter) {
        if(occupants.empty()) {
            return;
        }
        exitingThisUpdate.insert(occupants.front());
        occupants.erase(std::begin(occupants));
    }
}

StageProxy NotifiableQueue::Proxy(Simulation* simulation)
{
    return NotifiableQueueProxy(simulation, this);
}

const std::vector<GenericAgent::ID>& NotifiableQueue::Occupants() const
{
    return occupants;
}

void NotifiableQueue::Update(const EnvironmentQuery& envQuery)
{
    const auto count_occupants = occupants.size();
    if(count_occupants == slots.size()) {
        return;
    }

    for(size_t index = count_occupants; index < slots.size(); ++index) {
        const auto& slot = slots[index].anchor;
        auto candidates = envQuery.AgentsInRange(slot.xy(), 2, [&](const GenericAgent& candidate) {
            return envQuery.NoGeometryBetween(slot, candidate.location);
        });

        GenericAgent::ID occupant = GenericAgent::ID::Invalid;
        double min_distance = std::numeric_limits<double>::max();
        for(const auto& agent : candidates) {
            if(agent.stageId != id || Contains(occupants, agent.id) ||
               exitingThisUpdate.contains(agent.id)) {
                continue;
            }
            const auto distance = (agent.location.xy() - slot.xy()).Norm();
            if(distance < min_distance) {
                min_distance = distance;
                occupant = agent.id;
            }
        }
        if(occupant != GenericAgent::ID::Invalid) {
            occupants.emplace_back(occupant);
        } else {
            return;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// DirectSteering
////////////////////////////////////////////////////////////////////////////////
StageTarget DirectSteering::Target(const GenericAgent& agent)
{
    return agent.finalTarget;
}
