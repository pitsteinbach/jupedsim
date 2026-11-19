// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "kernel.hpp"

#include <fmt/core.h>

#include <variant>

/// The footprint an agent occupies, described around its centre.
///
/// Operational models use it to measure gaps between agents perimeter to perimeter instead of
/// centre to centre. Every shape answers one question: how far is the perimeter from the centre in
/// a given direction. That is all perimeter_distance() needs, so new shapes (e.g. ellipses) only
/// have to implement centre_to_perimeter().
class AgentShape
{
public:
    virtual ~AgentShape() = default;

    /// Distance from the centre to the perimeter along @p direction.
    ///
    /// @param direction unit vector in world coordinates; zero if the direction is undefined, e.g.
    ///        two agents sharing a centre. Shapes must still return a valid distance then.
    /// @param orientation unit vector in world coordinates the agent faces; shapes without
    ///        rotational symmetry measure @p direction relative to it.
    virtual double
    centre_to_perimeter(const Vector& direction, const Vector& orientation) const = 0;
};

class CircleShape : public AgentShape
{
    double _radius;

public:
    explicit CircleShape(double radius) : _radius(radius) {}

    double radius() const { return _radius; }

    double centre_to_perimeter(const Vector&, const Vector&) const override { return _radius; }
};

/// Ellipse with one semi-axis along the orientation and one across it.
class EllipseShape : public AgentShape
{
    double _semi_axis_along;
    double _semi_axis_across;

public:
    EllipseShape(double semi_axis_along, double semi_axis_across)
        : _semi_axis_along(semi_axis_along), _semi_axis_across(semi_axis_across)
    {
    }

    double semi_axis_along() const { return _semi_axis_along; }
    double semi_axis_across() const { return _semi_axis_across; }

    /// Polar radius of the ellipse at the angle between @p direction and @p orientation. Without
    /// either, the ellipse could face anywhere, so it returns the larger semi-axis.
    double centre_to_perimeter(const Vector& direction, const Vector& orientation) const override;
};

/// Every concrete shape, held by value so agent states stay copyable.
using AnyAgentShape = std::variant<CircleShape, EllipseShape>;

inline const AgentShape& as_agent_shape(const AnyAgentShape& shape)
{
    return std::visit([](const AgentShape& s) -> const AgentShape& { return s; }, shape);
}

/// Gap between the perimeters of two agents along the line connecting their centres.
///
/// @return centre distance minus both centre-to-perimeter distances; negative if the shapes overlap
double perimeter_distance(
    const AgentShape& first,
    const Point& centre_first,
    const Vector& orientation_first,
    const AgentShape& second,
    const Point& centre_second,
    const Vector& orientation_second);

template <>
struct fmt::formatter<CircleShape> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const CircleShape& c, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "Circle[radius={}]", c.radius());
    }
};

template <>
struct fmt::formatter<EllipseShape> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const EllipseShape& e, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "Ellipse[semi_axis_along={}, semi_axis_across={}]",
            e.semi_axis_along(),
            e.semi_axis_across());
    }
};

template <>
struct fmt::formatter<AnyAgentShape> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const AnyAgentShape& shape, FormatContext& ctx) const
    {
        return std::visit(
            [&ctx](const auto& s) { return fmt::format_to(ctx.out(), "{}", s); }, shape);
    }
};
