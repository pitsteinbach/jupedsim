// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "CfgCgal.hpp"
#include "Point.hpp"

#include <limits>
#include <span>
#include <vector>

class Router
{
public:
    /// Sentinel returned by DestinationId() for stages that route to a raw point (DirectSteering,
    /// WaitingSet, Queue) rather than a pre-registered polygon destination.
    static constexpr size_t DirectSteeringId = std::numeric_limits<size_t>::max();

    virtual ~Router() = default;

    Router() = default;
    Router(const Router&) = delete;
    Router& operator=(const Router&) = delete;
    Router(Router&&) = delete;
    Router& operator=(Router&&) = delete;

    /// Register a polygon destination (e.g. an exit area) with the router.
    /// Returns an opaque ID that can be passed to the ID-based compute methods.
    virtual size_t AddDestination(const Poly& area) = 0;

    /// Register a collection of polygons as a single merged destination.
    /// All cells inside any of the polygons become eikonal sources for one shared solve,
    /// so agents route to the nearest polygon automatically.
    virtual size_t AddDestination(std::span<const Poly> areas) = 0;

    /// Route to a pre-registered polygon destination by ID.
    virtual Point ComputeWaypoint(Point currentPosition, size_t destinationId) = 0;
    virtual std::vector<Point> ComputeAllWaypoints(Point currentPosition, size_t destinationId) = 0;

    /// Route to a raw point — used by DirectSteering, WaitingSet, Queue stages.
    virtual Point ComputeWaypoint(Point currentPosition, Point destination) = 0;
    virtual std::vector<Point> ComputeAllWaypoints(Point currentPosition, Point destination) = 0;

    /// Returns true if p lies inside the routable domain.
    virtual bool IsRoutable(Point p) const = 0;

    /// Called once per simulation tick with current agent positions. Implementations that
    /// support density-based routing use this to recount agents and trigger periodic
    /// speed-field rebuilds. The default is a no-op so non-density routers need not override it.
    virtual void UpdateDensity(std::span<const Point> /*positions*/) {}

    /// Called after UpdateDensity with the unique set of destinations that will be queried
    /// during the coming tactical phase. Implementations may compute those floor fields in
    /// parallel here so the per-agent routing loop only reads cached results. The default
    /// is a no-op; IDs must be unique across the two spans (no duplicates within ids).
    virtual void
    PrecomputeDestinations(std::span<const size_t> /*ids*/, std::span<const Point> /*points*/)
    {
    }

    /// Called once per simulation tick; implementations may use this to update internal state.
    virtual void Update() = 0;
};
