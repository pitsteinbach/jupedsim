// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "agent_shape.hpp"
#include "kernel.hpp"

#include <fmt/core.h>

struct CollisionFreeSpeedModelState {
    Vector orientation{0.0, 0.0};
    double time_gap{1};
    double v0{1.2};
    AnyAgentShape shape{CircleShape{0.2}};
};

template <>
struct fmt::formatter<CollisionFreeSpeedModelState> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const CollisionFreeSpeedModelState& m, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "CollisionFreeSpeedModel[orientation={}, timeGap={}, v0={}, shape={}])",
            m.orientation,
            m.time_gap,
            m.v0,
            m.shape);
    }
};
