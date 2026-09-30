// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Geometry/Geometry.hpp"
#include "Geometry/WalkableSurface.hpp"
#include "GeometryFixtures.hpp"
#include "HybridRoutingEngine.hpp"
#include "Journey.hpp"
#include "OperationalModels/CollisionFreeSpeedModel/CollisionFreeSpeedModel.hpp"
#include "Polygon.hpp"
#include "Simulation.hpp"
#include "SimulationError.hpp"
#include "StageDescription.hpp"
#include "SurfaceMeshShortestPathRoutingEngine.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

namespace
{
/// Two 10 x 10 rooms on one level, joined along their full height by a 1 m wide flat
/// connector (x in [10, 11]): a 10 m wide doorway.
std::unique_ptr<Geometry> two_rooms()
{
    WalkableSurface surface{};
    const auto a = surface.AddRegion({{{0, 0}, {10, 0}, {10, 10}, {0, 10}}, {}}, 0.0);
    const auto b = surface.AddRegion({{{11, 0}, {21, 0}, {21, 10}, {11, 10}}, {}}, 0.0);
    surface.ConnectRegions(a, {{10, 0}, {10, 10}}, b, {{11, 0}, {11, 10}});
    return surface.CreateGeometry();
}

Poly rect(double x0, double y0, double x1, double y1)
{
    return Polygon{std::vector<Point>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
}
} // namespace

TEST(HybridRoutingEngine, PointsFollowTheGeodesic)
{
    const auto geo = two_rooms();
    HybridRoutingEngine hybrid{*geo};
    SurfaceMeshShortestPathRoutingEngine geodesic{*geo};

    const Point3D from{2, 2, 0};
    const Point3D to{19, 8, 0};
    EXPECT_EQ(hybrid.GetShortestPath(from, to), geodesic.GetShortestPath(from, to));
}

TEST(HybridRoutingEngine, IdsAreHandedOutOnceAndChecked)
{
    const auto geo = two_rooms();
    HybridRoutingEngine engine{*geo};

    const auto p = engine.AddDestination(Point3D{19, 8, 0});
    EXPECT_EQ(engine.AddDestination(Point3D{19, 8, 0}), p) << "same point, same id";

    const auto pieces = geo->split_into_region_pieces(rect(19, 0, 21, 10), 0.0);
    const auto a1 = engine.AddDestination(std::span<const AreaPiece>{pieces});
    const auto a2 = engine.AddDestination(std::span<const AreaPiece>{pieces});
    EXPECT_NE(a1, a2) << "every area is its own destination";
    EXPECT_NE(a1, p);

    for(const auto id : {p, a1, a2}) {
        EXPECT_TRUE(engine.HasDestination(id));
    }
    EXPECT_FALSE(engine.HasDestination(DestinationId{}));
    EXPECT_FALSE(engine.HasDestination(DestinationId{42}));

    const auto from = geo->get_location(2, 2, 0);
    ASSERT_TRUE(from);
    EXPECT_THROW(engine.ComputeWaypoint(*from, DestinationId{42}), SimulationError);
}

TEST(HybridRoutingEngine, AnAreaAcrossSeamsIsSplitPerRegion)
{
    const auto geo = two_rooms();
    // x in [9, 12]: the end of room a, the whole connector, the start of room b.
    const auto pieces = geo->split_into_region_pieces(rect(9, 2, 12, 8), 0.0);

    std::set<std::size_t> regions{};
    double area = 0.0;
    for(const auto& piece : pieces) {
        regions.insert(piece.region);
        area += CGAL::to_double(piece.polygon.outer_boundary().area());
        EXPECT_EQ(geo->anchor_of(piece).region(), piece.region);
    }
    EXPECT_EQ(regions.size(), 3u);
    EXPECT_NEAR(area, 3.0 * 6.0, 1e-9);

    HybridRoutingEngine engine{*geo};
    const auto exit = engine.AddDestination(std::span<const AreaPiece>{pieces});
    // Reached from either side, each walks towards its own side of the strip.
    const auto west = geo->get_location(2, 5, 0);
    const auto east = geo->get_location(19, 5, 0);
    ASSERT_TRUE(west && east);
    EXPECT_GT(engine.ComputeWaypoint(*west, exit).x, 2.0);
    EXPECT_LT(engine.ComputeWaypoint(*east, exit).x, 19.0);
}

/// Without a 2D region graph an area is not cut: it is routed to the anchor of its one piece.
TEST(HybridRoutingEngine, MeshBuiltGeometriesRouteAreasToTheirAnchors)
{
    // Ground floor, a flight up 3 m, a landing: straight from a surface mesh.
    const auto geo = test_geometries::straight_stair_to_a_landing();
    ASSERT_EQ(geo->region_graph_2d(), nullptr);
    HybridRoutingEngine engine{*geo};

    const auto pieces = geo->split_into_region_pieces(rect(18, 2, 20, 6), 3.0);
    ASSERT_EQ(pieces.size(), 1u) << "nothing to cut along: one piece";
    const auto exit = engine.AddDestination(std::span<const AreaPiece>{pieces});

    const auto from = geo->get_location(2, 4, 0);
    ASSERT_TRUE(from);
    const auto path = engine.GetShortestPath(from->position_3d(), exit);
    ASSERT_FALSE(path.empty());
    // Ends at the anchor: the centre of the exit, up on the landing.
    EXPECT_NEAR(path.back().x(), 19.0, 1e-9);
    EXPECT_NEAR(path.back().y(), 4.0, 1e-9);
    EXPECT_NEAR(path.back().z(), 3.0, 1e-9);
}

TEST(HybridRoutingEngine, AgentsLeaveThroughAnExitAcrossASeam)
{
    auto sim = std::make_unique<Simulation>(
        std::make_unique<CollisionFreeSpeedModel>(8.0, 0.1, 5.0, 0.02), two_rooms(), 0.01);

    const auto exit = sim->AddStage(ExitDescription{Polygon{{{9, 0}, {12, 0}, {12, 10}, {9, 10}}}});
    const auto journey = sim->AddJourney({{exit, NonTransitionDescription{}}});
    sim->AddAgent(journey, exit, Point{2, 5}, CollisionFreeSpeedModel::State{}, 0.0);
    sim->AddAgent(journey, exit, Point{19, 5}, CollisionFreeSpeedModel::State{}, 0.0);

    for(int step = 0; step < 2000 && sim->AgentCount() > 0; ++step) {
        sim->Iterate();
    }
    EXPECT_EQ(sim->AgentCount(), 0u) << "not every agent made it into the exit";
}
