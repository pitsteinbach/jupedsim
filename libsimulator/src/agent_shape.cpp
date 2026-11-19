// SPDX-License-Identifier: LGPL-3.0-or-later
#include "agent_shape.hpp"

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
    return centre_distance - first.centre_to_perimeter(first_to_second, orientation_first) -
           second.centre_to_perimeter(-first_to_second, orientation_second);
}
