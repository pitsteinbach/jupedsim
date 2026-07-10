// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "FormatAny.hpp"

#include <any>
#include <type_traits>
#include <utility>

/// Type-erased per-agent state for custom (Python) operational models.
///
/// CustomModelState is stored in AgentState::extras so that built-in C++ models can
/// interact with Python custom model agents as neighbors — accessing position, radius,
/// and velocity via the common AgentState fields without touching the Python payload.
///
/// The payload type used by the Python binding layer is GilSafePyObject; libsimulator
/// itself treats it as opaque via std::any.
///
/// Payload types must be copy-constructible. GenericAgent values are copied by
/// neighborhood queries, so custom state must remain valid under value-copy semantics.
///
/// Access is runtime-typed: Get<T>() must use the exact stored type T. A type mismatch
/// throws std::bad_any_cast.
class CustomModelState
{
private:
    std::any value{};
    FormatFn format{};

public:
    template <typename T>
        requires(!std::is_same_v<std::decay_t<T>, CustomModelState>)
    explicit CustomModelState(T&& value)
        : value(std::forward<T>(value)), format(makeFormatFn<T>())
    {
        using Stored = std::decay_t<T>;
        static_assert(
            std::is_copy_constructible_v<Stored>,
            "CustomModelState payloads must be copy-constructible");
    }

    template <typename T>
    T& Get()
    {
        return std::any_cast<T&>(value);
    }

    template <typename T>
    const T& Get() const
    {
        return std::any_cast<const T&>(value);
    }

    template <typename T>
    void Set(T&& newValue)
    {
        using Stored = std::decay_t<T>;
        std::any_cast<Stored&>(value) = std::forward<T>(newValue);
    }

    friend struct fmt::formatter<CustomModelState>;
};

template <>
struct fmt::formatter<CustomModelState> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const CustomModelState& value, fmt::format_context& ctx) const
    {
        return value.format(value.value, ctx);
    }
};
