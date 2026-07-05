// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "AnticipationVelocityModel.hpp"
#include "CollisionFreeSpeedModel.hpp"
#include "CollisionFreeSpeedModelV2.hpp"
#include "CollisionFreeSpeedModelV3.hpp"
#include "GeneralizedCentrifugalForceModel.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/CustomModel/CustomModelData.hpp"
#include "OperationalModels/OperationalModelType.hpp"
#include "Point.hpp"
#include "SocialForceModel.hpp"
#include "UniqueID.hpp"
#include "Visitor.hpp"
#include "WarpDriver/WarpDriverModel.hpp"

#include <fmt/core.h>

#include <concepts>
#include <deque>
#include <utility>
#include <variant>
class Journey;
class BaseStage;

/// Agent position is owned by the per-model agent state. Every alternative of
/// GenericAgent::Model must satisfy this concept; the framework accesses the
/// position type-erased through Pos().
template <typename T>
concept ModelAgentState = requires(T t) {
    // Pos() hands out mutable Point& into the state, so a convertible or const member
    // is not enough.
    { t.position } -> std::same_as<Point&>;
};

template <typename Variant>
inline constexpr bool EachAlternativeIsModelAgentState = false;
template <typename... Ts>
inline constexpr bool EachAlternativeIsModelAgentState<std::variant<Ts...>> =
    (ModelAgentState<Ts> && ...);

struct GenericAgent;
const Point& Pos(const GenericAgent& agent);
Point& Pos(GenericAgent& agent);

struct GenericAgent {
    using ID = jps::UniqueID<GenericAgent>;
    ID id{};

    jps::UniqueID<Journey> journeyId{jps::UniqueID<Journey>::Invalid};
    jps::UniqueID<BaseStage> stageId{jps::UniqueID<BaseStage>::Invalid};

    // This is evaluated by the "operational level"
    Point destination{};
    Point target{};

    using Model = std::variant<
        GeneralizedCentrifugalForceModel::Agent,
        CollisionFreeSpeedModel::Agent,
        CollisionFreeSpeedModelV2::Agent,
        CollisionFreeSpeedModelV3::Agent,
        AnticipationVelocityModel::Agent,
        SocialForceModel::Agent,
        WarpDriverModel::Agent,
        CustomModelData>;
    static_assert(
        EachAlternativeIsModelAgentState<Model>,
        "Every agent model state must provide a 'Point position' member");
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
        // The model variant is initialized above, only then can the position
        // be written through it.
        Pos(*this) = pos_;
    }
};

inline const Point& Pos(const GenericAgent& agent)
{
    return std::visit([](const auto& m) -> const Point& { return m.position; }, agent.model);
}

inline Point& Pos(GenericAgent& agent)
{
    return std::visit([](auto& m) -> Point& { return m.position; }, agent.model);
}

/// Maps agent model data to the operational model type it belongs to. Kept
/// exhaustive on purpose: adding a model type will not compile until the
/// mapping is extended.
inline OperationalModelType ModelTypeOf(const GenericAgent::Model& model)
{
    return std::visit(
        overloaded{
            [](const GeneralizedCentrifugalForceModel::Agent&) {
                return OperationalModelType::GENERALIZED_CENTRIFUGAL_FORCE;
            },
            [](const CollisionFreeSpeedModel::Agent&) {
                return OperationalModelType::COLLISION_FREE_SPEED;
            },
            [](const CollisionFreeSpeedModelV2::Agent&) {
                return OperationalModelType::COLLISION_FREE_SPEED_V2;
            },
            [](const CollisionFreeSpeedModelV3::Agent&) {
                return OperationalModelType::COLLISION_FREE_SPEED_V3;
            },
            [](const AnticipationVelocityModel::Agent&) {
                return OperationalModelType::ANTICIPATION_VELOCITY_MODEL;
            },
            [](const SocialForceModel::Agent&) { return OperationalModelType::SOCIAL_FORCE; },
            [](const WarpDriverModel::Agent&) { return OperationalModelType::WARP_DRIVER; },
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
