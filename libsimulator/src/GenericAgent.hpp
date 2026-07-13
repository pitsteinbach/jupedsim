// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "AgentRouting.hpp"
#include "AgentState.hpp"
#include "OperationalModels/OperationalModelType.hpp"
#include "Point.hpp"
#include "UniqueID.hpp"

#include <fmt/core.h>

#include <deque>
#include <utility>

class Journey;
class BaseStage;

struct GenericAgent {
    using ID = jps::UniqueID<GenericAgent>;
    ID id{};

    jps::UniqueID<Journey> journeyId{jps::UniqueID<Journey>::Invalid};
    jps::UniqueID<BaseStage> stageId{jps::UniqueID<BaseStage>::Invalid};

    AgentRouting routing{};

    /// Unified per-agent model state. Built-in model agents leave customState empty;
    /// Python custom model agents store a GilSafePyObject in customState.
    using Model = AgentState;
    Model model{};

    GenericAgent(
        ID id_,
        jps::UniqueID<Journey> journeyId_,
        jps::UniqueID<BaseStage> stageId_,
        Point pos_,
        Model model_)
        : id(id_ != ID::Invalid ? id_ : ID{})
        , journeyId(journeyId_)
        , stageId(stageId_)
        , routing{{}, pos_}
        , model(std::move(model_))
    {
        model.position = pos_;
    }
};

inline const Point& Pos(const GenericAgent& agent)
{
    return agent.model.position;
}

inline Point& Pos(GenericAgent& agent)
{
    return agent.model.position;
}

inline OperationalModelType ModelTypeOf(const GenericAgent::Model& model)
{
    return model.type;
}

template <class Agent>
using AgentContainer = std::deque<Agent>;

template <>
struct fmt::formatter<GenericAgent> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const GenericAgent& agent, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "Agent[id={}, journey={}, stage={}, destination={}, waypoint={}, pos={}, "
            "model={})",
            agent.id,
            agent.journeyId,
            agent.stageId,
            agent.routing.destination,
            agent.routing.target,
            Pos(agent),
            agent.model);
    }
};
