// SPDX-License-Identifier: LGPL-3.0-or-later
#include "agent_shape.hpp"
#include "kernel.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace
{
/// Rectangle with half-extents along / across the orientation. Exercises the orientation and
/// direction arguments, which circles ignore.
class HalfExtentShape : public AgentShape
{
    double _along;
    double _across;

public:
    HalfExtentShape(double along, double across) : _along(along), _across(across) {}

    double centre_to_perimeter(const Vector& direction, const Vector& orientation) const override
    {
        const double cos = std::abs(direction.scalar_product(orientation));
        const double sin = std::abs(direction.cross_product(orientation));
        if(cos * _across >= sin * _along) {
            return cos > 0 ? _along / cos : _along;
        }
        return _across / sin;
    }
};
} // namespace

TEST(AgentShape, CircleIsRadiusInEveryDirection)
{
    const CircleShape circle{0.3};
    for(const Vector& direction : {Vector{1, 0}, Vector{0, -1}, Vector{0.6, 0.8}, Vector{}}) {
        ASSERT_DOUBLE_EQ(circle.centre_to_perimeter(direction, Vector{1, 0}), 0.3);
    }
}

TEST(AgentShape, PerimeterDistanceOfCircles)
{
    const CircleShape a{0.2};
    const CircleShape b{0.3};
    const Vector any{0, 1};
    ASSERT_DOUBLE_EQ(perimeter_distance(a, {0, 0}, any, b, {3, 4}, any), 4.5);
    ASSERT_DOUBLE_EQ(perimeter_distance(b, {3, 4}, any, a, {0, 0}, any), 4.5);
}

TEST(AgentShape, PerimeterDistanceIsNegativeOnOverlap)
{
    const CircleShape a{0.2};
    const Vector any{1, 0};
    ASSERT_DOUBLE_EQ(perimeter_distance(a, {0, 0}, any, a, {0.3, 0}, any), -0.1);
    ASSERT_DOUBLE_EQ(perimeter_distance(a, {1, 1}, any, a, {1, 1}, any), -0.4);
}

TEST(AgentShape, PerimeterDistanceUsesEachOrientation)
{
    const HalfExtentShape shape{0.5, 0.1};
    const Point left{0, 0};
    const Point right{2, 0};
    const Vector along_x{1, 0};
    const Vector along_y{0, 1};

    // Facing each other: both long half-extents lie on the centre line.
    ASSERT_DOUBLE_EQ(perimeter_distance(shape, left, along_x, shape, right, -along_x), 1.0);
    // Side by side: both short half-extents.
    ASSERT_DOUBLE_EQ(perimeter_distance(shape, left, along_y, shape, right, along_y), 1.8);
    // Mixed: one long, one short.
    ASSERT_DOUBLE_EQ(perimeter_distance(shape, left, along_x, shape, right, along_y), 1.4);
}

TEST(AgentShape, EllipseReachesItsSemiAxesAlongAndAcrossTheOrientation)
{
    const EllipseShape ellipse{0.4, 0.2};
    const Vector orientation{0.6, 0.8};
    ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(orientation, orientation), 0.4);
    ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(-orientation, orientation), 0.4);
    ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(orientation.rotate90_deg(), orientation), 0.2);
    ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(-orientation.rotate90_deg(), orientation), 0.2);
}

TEST(AgentShape, EllipseReachIsItsPolarRadius)
{
    const double a = 0.4;
    const double b = 0.2;
    const EllipseShape ellipse{a, b};
    const Vector orientation{0, 1};
    for(const double angle : {0.1, 0.5, 1.0, 2.0, 3.0, -1.2}) {
        const Vector local{std::cos(angle), std::sin(angle)};
        const Vector direction = local.rotate(orientation.x(), orientation.y());
        const double reach = ellipse.centre_to_perimeter(direction, orientation);
        // The perimeter point in the ellipse's own frame satisfies x²/a² + y²/b² = 1.
        const Vector perimeter = local * reach;
        ASSERT_NEAR(
            perimeter.x() * perimeter.x() / (a * a) + perimeter.y() * perimeter.y() / (b * b),
            1.0,
            1e-12);
    }
}

TEST(AgentShape, EllipseWithEqualAxesIsACircle)
{
    const EllipseShape ellipse{0.3, 0.3};
    for(const Vector& direction : {Vector{1, 0}, Vector{0, -1}, Vector{0.6, 0.8}}) {
        ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(direction, Vector{0.8, -0.6}), 0.3);
    }
}

TEST(AgentShape, EllipseWithoutDirectionOrOrientationReachesItsLargerSemiAxis)
{
    const EllipseShape ellipse{0.2, 0.4};
    ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(Vector{}, Vector{1, 0}), 0.4);
    ASSERT_DOUBLE_EQ(ellipse.centre_to_perimeter(Vector{1, 0}, Vector{}), 0.4);
}

TEST(AgentShape, PerimeterDistanceOfEllipsesUsesEachOrientation)
{
    const EllipseShape ellipse{0.4, 0.2};
    const Point left{0, 0};
    const Point right{2, 0};
    const Vector along_x{1, 0};
    const Vector along_y{0, 1};

    ASSERT_DOUBLE_EQ(perimeter_distance(ellipse, left, along_x, ellipse, right, -along_x), 1.2);
    ASSERT_DOUBLE_EQ(perimeter_distance(ellipse, left, along_y, ellipse, right, along_y), 1.6);
    ASSERT_DOUBLE_EQ(perimeter_distance(ellipse, left, along_x, ellipse, right, along_y), 1.4);
}

TEST(AgentShape, VariantDispatchesToTheHeldShape)
{
    const AnyAgentShape circle{CircleShape{0.3}};
    const AnyAgentShape ellipse{EllipseShape{0.4, 0.2}};
    const Vector across{0, 1};
    const Vector orientation{1, 0};
    ASSERT_DOUBLE_EQ(as_agent_shape(circle).centre_to_perimeter(across, orientation), 0.3);
    ASSERT_DOUBLE_EQ(as_agent_shape(ellipse).centre_to_perimeter(across, orientation), 0.2);
    ASSERT_EQ(fmt::format("{}", circle), "Circle[radius=0.3]");
    ASSERT_EQ(fmt::format("{}", ellipse), "Ellipse[semi_axis_along=0.4, semi_axis_across=0.2]");
}
