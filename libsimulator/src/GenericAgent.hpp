// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "AgentState.hpp"
#include "OperationalModels/CustomModel/CustomModelData.hpp"
#include "OperationalModels/OperationalModelType.hpp"
#include "Point.hpp"
#include "UniqueID.hpp"
#include "Visitor.hpp"

#include <fmt/core.h>

#include <deque>
#include <utility>
#include <variant>

class Journey;
class BaseStage;

struct GenericAgent;
const Point& Pos(const GenericAgent& agent);
Point& Pos(GenericAgent& agent);

struct GenericAgent {
    using ID = jps::UniqueID<GenericAgent>;
    ID id{};

    jps::UniqueID<Journey> journeyId{jps::UniqueID<Journey>::Invalid};
    jps::UniqueID<BaseStage> stageId{jps::UniqueID<BaseStage>::Invalid};

    Point destination{};
    Point target{};

    /// Built-in models store an AgentState; custom models store CustomModelData.
    using Model = std::variant<AgentState, CustomModelData>;
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
        , target(pos_)
        , model(std::move(model_))
    {
        Pos(*this) = pos_;
    }
};

inline const Point& Pos(const GenericAgent& agent)
{
    return std::visit(
        overloaded{
            [](const AgentState& s) -> const Point& { return s.position; },
            [](const CustomModelData& d) -> const Point& { return d.position; }},
        agent.model);
}

inline Point& Pos(GenericAgent& agent)
{
    return std::visit(
        overloaded{
            [](AgentState& s) -> Point& { return s.position; },
            [](CustomModelData& d) -> Point& { return d.position; }},
        agent.model);
}

/// Returns the operational model type of the agent.
inline OperationalModelType ModelTypeOf(const GenericAgent::Model& model)
{
    return std::visit(
        overloaded{
            [](const AgentState& s) { return s.type; },
            [](const CustomModelData&) { return OperationalModelType::CUSTOM_MODEL; }},
        model);
}

template <class Agent>
using AgentContainer = std::deque<Agent>;

template <>
struct fmt::formatter<GenericAgent> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const GenericAgent& agent, FormatContext& ctx) const
    {
        return std::visit(
            [&ctx, &agent](const auto& m) {
                return fmt::format_to(
                    ctx.out(),
                    "Agent[id={}, journey={}, stage={}, destination={}, waypoint={}, pos={}, "
                    "model={})",
                    agent.id,
                    agent.journeyId,
                    agent.stageId,
                    agent.destination,
                    agent.target,
                    Pos(agent),
                    m);
            },
            agent.model);
    }
};
