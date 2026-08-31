// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "CfgCgal.hpp"
#include "Mesh.hpp"
#include "Point.hpp"
#include "Routing.hpp"

#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

class RoutingEngine : public Router
{
    CDT cdt{};
    std::unique_ptr<Mesh> mesh{};
    std::vector<Point> _destinations;

public:
    RoutingEngine();
    explicit RoutingEngine(const PolyWithHoles& poly);
    ~RoutingEngine() override = default;

    RoutingEngine(const RoutingEngine& other) = delete;
    RoutingEngine& operator=(const RoutingEngine& other) = delete;

    RoutingEngine(RoutingEngine&& other) = delete;
    RoutingEngine& operator=(RoutingEngine&& other) = delete;

    size_t AddDestination(const Poly& area) override;
    size_t AddDestination(std::span<const Poly> areas) override;

    Point ComputeWaypoint(Point currentPosition, size_t destinationId) override;
    std::vector<Point> ComputeAllWaypoints(Point currentPosition, size_t destinationId) override;

    Point ComputeWaypoint(Point currentPosition, Point destination) override;
    std::vector<Point> ComputeAllWaypoints(Point currentPosition, Point destination) override;
    bool IsRoutable(Point p) const override;
    void Update() override;

    const Mesh* MeshData() const { return mesh.get(); }

private:
    CDT::Face_handle find_face(K::Point_2) const;
    std::vector<Point>
    straightenPath(Point from, Point to, const std::vector<CDT::Face_handle>& path);
};
