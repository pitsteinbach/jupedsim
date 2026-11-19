// SPDX-License-Identifier: LGPL-3.0-or-later
#include "agent_shape.hpp"

#include <algorithm>
#include <cmath>

double EllipseShape::centre_to_perimeter(const Vector& direction, const Vector& orientation) const
{
    if(direction.is_zero_length() || orientation.is_zero_length()) {
        return std::max(_semi_axis_along, _semi_axis_across);
    }
    // Components of the direction in the ellipse's own frame, x along the orientation.
    const double along = direction.scalar_product(orientation);
    const double across = orientation.cross_product(direction);
    const double a = _semi_axis_along;
    const double b = _semi_axis_across;
    return a * b / std::hypot(b * along, a * across);
}

double perimeter_distance(
    const AgentShape& first,
    const Point& centre_first,
    const Vector& orientation_first,
    const AgentShape& second,
    const Point& centre_second,
    const Vector& orientation_second)
{
    const auto [centre_distance, first_to_second] =
        (centre_second - centre_first).norm_and_normalized();
    // Summed first, as models did with radii, so circles reproduce their trajectories bit for bit.
    const double contact_distance =
        first.centre_to_perimeter(first_to_second, orientation_first) +
        second.centre_to_perimeter(-first_to_second, orientation_second);
    return centre_distance - contact_distance;
}
