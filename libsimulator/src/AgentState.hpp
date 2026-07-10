// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "OperationalModelType.hpp"
#include "OperationalModels/CustomModel/CustomModelState.hpp"
#include "Point.hpp"

#include <fmt/core.h>

#include <optional>
#include <variant>

/// Model-specific per-agent state for GeneralizedCentrifugalForceModel.
struct GCFMExtras {
    double speed{};
    Point e0{};
    int orientationDelay{};
    double Av{1.0};
    double AMin{0.2};
    double BMin{0.2};
    double BMax{0.4};
    double maxNeighborInteractionDistance{2};
    double maxGeometryInteractionDistance{2};
    double maxNeighborInterpolationDistance{0.1};
    double maxGeometryInterpolationDistance{0.1};
    double maxNeighborRepulsionForce{9};
    double maxGeometryRepulsionForce{3};
};

/// Model-specific per-agent state for SocialForceModel.
struct SFMExtras {
    double agentScale{2000.0};
    double obstacleScale{2000.0};
    double forceDistance{0.08};
    double bodyForce{120000};
    double friction{240000};
};

/// Model-specific per-agent state for AnticipationVelocityModel.
struct AVMExtras {
    double wallBufferDistance{0.1};
    double anticipationTime{1.0};
    double pushoutStrength{0.3};
};

/// Model-specific per-agent state for CollisionFreeSpeedModelV3.
struct CFSv3Extras {
    double rangeXScale{20.0};
    double rangeYScale{8.0};
    double thetaMaxUpperBound{1.57};
    double agentBuffer{0.0};
    double headingAngle{0.0};
};

/// Model-specific per-agent state for WarpDriverModel.
struct WarpExtras {
    double stuckTime{0.0};
    double anchorX{0.0};
    double anchorY{0.0};
    double detourTime{0.0};
    int detourSide{1};
    double timeHorizon{2.0};
    double stepSize{0.5};
    double timeUncertainty{0.5};
    double velocityUncertaintyX{0.2};
    double velocityUncertaintyY{0.2};
    int numSamples{20};
};

/// Unified per-agent state for all operational models (built-in and Python custom).
///
/// Common fields are std::optional<T>: nullopt means "use the model's own default".
/// The active model reads a field via field.value_or(Model::Defaults::fieldName).
/// Model-specific state lives in the Extras variant:
///   - nullopt:          CFS, CFSv2 (fully covered by common fields)
///   - GCFMExtras:       GeneralizedCentrifugalForceModel
///   - SFMExtras:        SocialForceModel
///   - AVMExtras:        AnticipationVelocityModel
///   - CFSv3Extras:      CollisionFreeSpeedModelV3
///   - WarpExtras:       WarpDriverModel
///   - CustomModelState: Python custom model (payload is a GilSafePyObject)
///
/// On a model switch: retain the existing common optionals (preserving any explicit
/// user values) and replace extras with the new model's Extras struct.
struct AgentState {
    OperationalModelType type{OperationalModelType::COLLISION_FREE_SPEED};

    /// Position is always present; all models must keep it current.
    Point position{};

    // ---- common optional fields — nullopt means "use model default" ----
    std::optional<Point> orientation{};
    std::optional<double> v0{};
    std::optional<double> radius{};
    std::optional<double> timeGap{};
    std::optional<double> strengthNeighborRepulsion{};
    std::optional<double> rangeNeighborRepulsion{};
    std::optional<double> strengthGeometryRepulsion{};
    std::optional<double> rangeGeometryRepulsion{};
    std::optional<Point> velocity{}; // AVM, SFM
    std::optional<double> mass{}; // GCFM, SFM
    std::optional<double> reactionTime{}; // AVM, SFM; maps to tau in GCFM

    // ---- model-specific extras ----
    using Extras = std::variant<
        GCFMExtras,
        SFMExtras,
        AVMExtras,
        CFSv3Extras,
        WarpExtras,
        CustomModelState>;
    std::optional<Extras> extras{};
};

template <>
struct fmt::formatter<AgentState> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const AgentState& s, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "AgentState[type={}, pos={}, orientation={}, v0={}, radius={}]",
            ToString(s.type),
            s.position,
            s.orientation ? fmt::format("{}", *s.orientation) : "default",
            s.v0 ? fmt::format("{}", *s.v0) : "default",
            s.radius ? fmt::format("{}", *s.radius) : "default");
    }
};
