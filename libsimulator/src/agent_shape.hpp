// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "kernel.hpp"

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
