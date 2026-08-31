// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "Point.hpp"
#include "Polygon.hpp"

#include <variant>
#include <vector>

struct DirectSteeringDescription {
};

struct WaypointDescription {
    Point position;
    double distance;
};

struct ExitDescription {
    Polygon polygon;
};

struct MultiExitDescription {
    std::vector<Polygon> polygons;
};

struct NotifiableWaitingSetDescription {
    std::vector<Point> slots;
};

struct NotifiableQueueDescription {
    std::vector<Point> slots;
};

using StageDescription = std::variant<
    DirectSteeringDescription,
    WaypointDescription,
    ExitDescription,
    MultiExitDescription,
    NotifiableWaitingSetDescription,
    NotifiableQueueDescription>;
