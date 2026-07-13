// SPDX-License-Identifier: LGPL-3.0-or-later
#include "GeneralizedCentrifugalForceModel.hpp"

#include "AgentState.hpp"
#include "Ellipse.hpp"
#include "GenericAgent.hpp"
#include "Macros.hpp"
#include "Mathematics.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Simulation.hpp"
#include "SimulationError.hpp"

#include <Logger.hpp>

#include <optional>
#include <stdexcept>

AgentState GeneralizedCentrifugalForceModel::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::GENERALIZED_CENTRIFUGAL_FORCE,
        .position = pos,
        .orientation = Point{1.0, 0.0},
        .v0 = Defaults::v0,
        .strengthNeighborRepulsion = Defaults::strengthNeighborRepulsion,
        .strengthGeometryRepulsion = Defaults::strengthGeometryRepulsion,
        .mass = Defaults::mass,
        .reactionTime = Defaults::reactionTime,
        .extras = GCFMExtras{
            .speed = 0.0,
            .e0 = Point{},
            .orientationDelay = 0,
            .Av = Defaults::Av,
            .AMin = Defaults::AMin,
            .BMin = Defaults::BMin,
            .BMax = Defaults::BMax,
            .maxNeighborInteractionDistance = Defaults::maxNeighborInteractionDistance,
            .maxGeometryInteractionDistance = Defaults::maxGeometryInteractionDistance,
            .maxNeighborInterpolationDistance = Defaults::maxNeighborInterpolationDistance,
            .maxGeometryInterpolationDistance = Defaults::maxGeometryInterpolationDistance,
            .maxNeighborRepulsionForce = Defaults::maxNeighborRepulsionForce,
            .maxGeometryRepulsionForce = Defaults::maxGeometryRepulsionForce,
        },
    };
}

OperationalModelType GeneralizedCentrifugalForceModel::Type() const
{
    return OperationalModelType::GENERALIZED_CENTRIFUGAL_FORCE;
}

void GeneralizedCentrifugalForceModel::ComputeNext(
    double dT,
    const AgentState& current,
    AgentState& next,
    const AgentRouting& routing,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    const auto neighborhood = neighborhoodSearch.GetNeighboringAgents(current.position, _cutOffRadius);
    const auto p1 = current.position;
    Point F_rep;
    for(const auto& neighbor : neighborhood) {
        if(Pos(neighbor) == current.position) {
            continue;
        }
        if(!geometry.IntersectsAny(LineSegment(p1, Pos(neighbor)))) {
            F_rep += ForceRepPed(current, neighbor);
        }
    }

    Point e0{};
    std::optional<Point> position{};
    std::optional<Point> velocity{};
    Point repwall = ForceRepRoom(current, geometry);
    const auto& extras = std::get<GCFMExtras>(*current.extras);
    const auto mass = current.mass.value_or(Defaults::mass);
    const auto tau = current.reactionTime.value_or(Defaults::reactionTime);
    Point fd = ForceDriv(current, routing.destination, mass, tau, dT, e0);
    Point acc = (fd + F_rep + repwall) / mass;

    velocity = (current.orientation.value_or(Point{1.0, 0.0}) * extras.speed) + acc * dT;
    position = current.position + *velocity * dT;

    auto& nextExtras = std::get<GCFMExtras>(*next.extras);
    nextExtras.e0 = e0;
    ++nextExtras.orientationDelay;
    if(position) {
        next.position = *position;
    }
    if(velocity) {
        next.orientation = (*velocity).Normalized();
        nextExtras.speed = (*velocity).Norm();
    }
}

void GeneralizedCentrifugalForceModel::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    const auto& state = agent.model;
    const auto& extras = std::get<GCFMExtras>(*state.extras);
    const auto orientation = state.orientation.value_or(Point{1.0, 0.0});

    if(!orientation.IsUnitLength()) {
        throw SimulationError("Orientation is invalid: {}. Length should be 1.", orientation);
    }

    validateConstraint(state.mass.value_or(Defaults::mass), 1.0, 100.0, "mass");
    validateConstraint(state.reactionTime.value_or(Defaults::reactionTime), 0.1, 10.0, "tau");
    validateConstraint(state.v0.value_or(Defaults::v0), 0.0, 10.0, "v0");
    validateConstraint(extras.Av, 0.0, 10.0, "Av");
    validateConstraint(extras.AMin, 0.1, 1.0, "AMin");
    validateConstraint(extras.BMin, 0.1, 1.0, "BMin");
    validateConstraint(extras.BMax, extras.BMin, 2.0, "BMax");

    const auto neighbors = neighborhoodSearch.GetNeighboringAgents(Pos(agent), 2);
    for(const auto& neighbor : neighbors) {
        if(agent.id == neighbor.id) {
            continue;
        }
        const auto contanctDist = AgentToAgentSpacing(agent, neighbor);
        const auto distance = (Pos(agent) - Pos(neighbor)).Norm();
        if(contanctDist >= distance) {
            throw SimulationError(
                "Model constraint violation: Agent {} too close to agent {}: distance {}, "
                "contactDist {}, "
                "effective distance {}",
                Pos(agent),
                Pos(neighbor),
                distance,
                contanctDist,
                distance - contanctDist);
        }
    }

    const auto maxRadius = std::max(extras.AMin, extras.BMax) / 2.;
    const auto lineSegments = geometry.LineSegmentsInDistanceTo(maxRadius, Pos(agent));
    if(std::begin(lineSegments) != std::end(lineSegments)) {
        throw SimulationError(
            "Model constraint violation: Agent {} too close to geometry boundaries, distance <= {}",
            Pos(agent),
            maxRadius);
    }
}

Point GeneralizedCentrifugalForceModel::ForceDriv(
    const AgentState& ped,
    Point target,
    double mass,
    double tau,
    double deltaT,
    Point& e0update) const
{
    Point F_driv;
    const auto pos = ped.position;
    const auto dist = (target - pos).Norm();
    const auto& extras = std::get<GCFMExtras>(*ped.extras);
    const auto orientation = ped.orientation.value_or(Point{1.0, 0.0});
    const auto v0 = ped.v0.value_or(Defaults::v0);
    if(dist > J_EPS_GOAL) {
        const Point e0 = mollify_e0(target, pos, deltaT, extras.orientationDelay, extras.e0);
        e0update = e0;
        F_driv = ((e0 * v0 - (orientation * extras.speed)) * mass) / tau;
    } else {
        const Point e0 = extras.e0;
        F_driv = ((e0 * v0 - (orientation * extras.speed)) * mass) / tau;
    }
    return F_driv;
}

Point GeneralizedCentrifugalForceModel::ForceRepPed(
    const AgentState& ped1,
    const GenericAgent& ped2agent) const
{
    const auto& s2 = ped2agent.model;
    const auto& e1 = std::get<GCFMExtras>(*ped1.extras);
    const auto& e2 = std::get<GCFMExtras>(*s2.extras);
    const auto orientation1 = ped1.orientation.value_or(Point{1.0, 0.0});
    const auto orientation2 = s2.orientation.value_or(Point{1.0, 0.0});
    const auto v0_1 = ped1.v0.value_or(Defaults::v0);
    const auto strengthN = ped1.strengthNeighborRepulsion.value_or(Defaults::strengthNeighborRepulsion);
    const auto agent1_mass = ped1.mass.value_or(Defaults::mass);

    Point F_rep;
    Point distp12 = Pos(ped2agent) - ped1.position;
    const Point vp1 = orientation1 * e1.speed;
    const Point vp2 = orientation2 * e2.speed;
    Point ep12;
    double tmp, tmp2;
    double v_ij;
    double K_ij;
    double nom;
    double px;
    const auto dist_eff = AgentToAgentSpacingFromState(ped1, ped2agent);

    if(dist_eff >= e1.maxNeighborInteractionDistance) {
        return Point(0.0, 0.0);
    }

    const double mindist = 0.5;
    const double dist_intpol_left = mindist + e1.maxNeighborInterpolationDistance;
    const double dist_intpol_right =
        e1.maxNeighborInteractionDistance - e1.maxNeighborInterpolationDistance;
    const double smax = mindist - e1.maxNeighborInterpolationDistance;
    double f = 0.0f;
    double f1 = 0.0f;

    if(distp12.Norm() >= J_EPS) {
        ep12 = distp12.Normalized();
    } else {
        LOG_WARNING(
            "Distance between two pedestrians is small ({}<{}). Force can not be calculated.",
            distp12.Norm(),
            J_EPS);
        return F_rep;
    }
    tmp = (vp1 - vp2).ScalarProduct(ep12);
    v_ij = 0.5 * (tmp + fabs(tmp));
    tmp2 = vp1.ScalarProduct(ep12);

    if(vp1.Norm() < J_EPS) {
        K_ij = 0;
    } else {
        double bla = tmp2 + fabs(tmp2);
        K_ij = 0.25 * bla * bla / vp1.ScalarProduct(vp1);
        if(K_ij < J_EPS * J_EPS) {
            return Point(0.0, 0.0);
        }
    }

    nom = strengthN * v0_1 + v_ij;
    nom *= nom;

    K_ij = sqrt(K_ij);
    if(dist_eff <= smax) {
        f = -agent1_mass * K_ij * nom / dist_intpol_left;
        F_rep = ep12 * e1.maxNeighborRepulsionForce * f;
        return F_rep;
    }

    if(dist_eff >= dist_intpol_right) {
        f = -agent1_mass * K_ij * nom / dist_intpol_right;
        f1 = -f / dist_intpol_right;
        px = hermite_interp(
            dist_eff, dist_intpol_right, e1.maxNeighborInteractionDistance, f, 0, f1, 0);
        F_rep = ep12 * px;
    } else if(dist_eff >= dist_intpol_left) {
        f = -agent1_mass * K_ij * nom / fabs(dist_eff);
        F_rep = ep12 * f;
    } else {
        f = -agent1_mass * K_ij * nom / dist_intpol_left;
        f1 = -f / dist_intpol_left;
        px = hermite_interp(
            dist_eff, smax, dist_intpol_left, e1.maxNeighborRepulsionForce * f, f, 0, f1);
        F_rep = ep12 * px;
    }
    if(F_rep.x != F_rep.x || F_rep.y != F_rep.y) {
        LOG_ERROR(
            "NAN return p1 pos={} p2 pos={} Frepx={:f} Frepy={:f} K_ij={:f}",
            ped1.position,
            Pos(ped2agent),
            F_rep.x,
            F_rep.y,
            K_ij);
    }
    return F_rep;
}

inline Point GeneralizedCentrifugalForceModel::ForceRepRoom(
    const AgentState& ped,
    const CollisionGeometry& geometry) const
{
    const auto& walls = geometry.LineSegmentsInApproxDistanceTo(ped.position);
    return std::accumulate(
        walls.cbegin(),
        walls.cend(),
        Point(0, 0),
        [this, &ped](const auto& acc, const auto& element) {
            return acc + ForceRepWall(ped, element);
        });
}

inline Point GeneralizedCentrifugalForceModel::ForceRepWall(
    const AgentState& ped,
    const LineSegment& w) const
{
    Point F = Point(0.0, 0.0);
    Point pt = w.ShortestPoint(ped.position);
    double wlen = w.LengthSquare();

    if(wlen < 0.01) {
        return F;
    }
    if(fabs((w.p1 - w.p2).ScalarProduct(ped.position - pt)) > J_EPS) {
        return F;
    }
    double mind = 0.5;
    const auto& extras = std::get<GCFMExtras>(*ped.extras);
    const auto orientation = ped.orientation.value_or(Point{1.0, 0.0});
    double vn = w.NormalComp(orientation * extras.speed);
    F = ForceRepStatPoint(ped, pt, mind, vn);
    return F;
}

Point GeneralizedCentrifugalForceModel::ForceRepStatPoint(
    const AgentState& ped,
    const Point& p,
    double l,
    double vn) const
{
    Point F_rep = Point(0.0, 0.0);
    const auto& extras = std::get<GCFMExtras>(*ped.extras);
    const auto orientation = ped.orientation.value_or(Point{1.0, 0.0});
    const auto v0 = ped.v0.value_or(Defaults::v0);
    const Point v = orientation * extras.speed;
    Point dist = p - ped.position;
    double d = dist.Norm();
    Point e_ij;

    double tmp;
    double bla;
    Point r;
    Point pinE;
    const Ellipse E{extras.Av, extras.AMin, extras.BMax, extras.BMin};

    if(d < J_EPS) {
        return Point(0.0, 0.0);
    }
    e_ij = dist / d;
    tmp = v.ScalarProduct(e_ij);
    bla = (tmp + fabs(tmp));
    if(!bla) {
        return Point(0.0, 0.0);
    }
    if(fabs(v.x) < J_EPS && fabs(v.y) < J_EPS) {
        return Point(0.0, 0.0);
    }
    double K_ij;
    K_ij = 0.5 * bla / v.Norm();
    pinE = p.TransformToEllipseCoordinates(ped.position, orientation.x, orientation.y);
    r = E.PointOnEllipse(pinE, extras.speed / v0, ped.position, extras.speed, orientation);
    F_rep = ForceInterpolation(ped, v0, K_ij, e_ij, vn, d, (r - ped.position).Norm(), l);
    return F_rep;
}

Point GeneralizedCentrifugalForceModel::ForceInterpolation(
    const AgentState& state,
    double v0,
    double K_ij,
    const Point& e,
    double vn,
    double d,
    double r,
    double l) const
{
    const auto& extras = std::get<GCFMExtras>(*state.extras);
    const auto strengthG = state.strengthGeometryRepulsion.value_or(Defaults::strengthGeometryRepulsion);
    Point F_rep;
    double nominator = strengthG * v0 + vn;
    nominator *= nominator * K_ij;
    double f = 0, f1 = 0;
    double smax = l - extras.maxGeometryInterpolationDistance;
    double dist_intpol_left = l + extras.maxGeometryInterpolationDistance;
    double dist_intpol_right =
        extras.maxGeometryInteractionDistance - extras.maxGeometryInterpolationDistance;

    double dist_eff = d - r;

    double px = 0;
    double tmp1 = extras.maxGeometryInteractionDistance;
    double tmp2 = dist_intpol_right;
    double tmp3 = dist_intpol_left;
    double tmp5 = smax + r;

    if(dist_eff >= tmp1) {
        return F_rep;
    }
    if(dist_eff <= tmp5) {
        F_rep = e * (-extras.maxGeometryRepulsionForce);
        return F_rep;
    }
    if(dist_eff > tmp2) {
        f = -nominator / dist_intpol_right;
        f1 = -f / dist_intpol_right;
        px = hermite_interp(
            dist_eff, dist_intpol_right, extras.maxGeometryInteractionDistance, f, 0, f1, 0);
        F_rep = e * px;
    } else if(dist_eff >= tmp3) {
        f = -nominator / fabs(dist_eff);
        F_rep = e * f;
    } else {
        f = -nominator / dist_intpol_left;
        f1 = -f / dist_intpol_left;
        px = hermite_interp(
            dist_eff, smax, dist_intpol_left, extras.maxGeometryRepulsionForce * f, f, 0, f1);
        F_rep = e * px;
    }
    return F_rep;
}

double GeneralizedCentrifugalForceModel::AgentToAgentSpacing(
    const GenericAgent& agent1,
    const GenericAgent& agent2) const
{
    const auto& s1 = agent1.model;
    const auto& s2 = agent2.model;
    const auto& e1 = std::get<GCFMExtras>(*s1.extras);
    const auto& e2 = std::get<GCFMExtras>(*s2.extras);
    const Ellipse E1{e1.Av, e1.AMin, e1.BMax, e1.BMin};
    const Ellipse E2{e2.Av, e2.AMin, e2.BMax, e2.BMin};
    const auto v0_1 = s1.v0.value_or(Defaults::v0);
    const auto v0_2 = s2.v0.value_or(Defaults::v0);
    const double scale1 = (v0_1 == 0.0) ? 1.0 : e1.speed / v0_1;
    const double scale2 = (v0_2 == 0.0) ? 1.0 : e2.speed / v0_2;
    const auto orientation1 = s1.orientation.value_or(Point{1.0, 0.0});
    const auto orientation2 = s2.orientation.value_or(Point{1.0, 0.0});

    return E1.EffectiveDistanceToEllipse(
        E2,
        Pos(agent1),
        Pos(agent2),
        scale1,
        scale2,
        e1.speed,
        e2.speed,
        orientation1,
        orientation2);
}

double GeneralizedCentrifugalForceModel::AgentToAgentSpacingFromState(
    const AgentState& agent1,
    const GenericAgent& agent2) const
{
    const auto& s2 = agent2.model;
    const auto& e1 = std::get<GCFMExtras>(*agent1.extras);
    const auto& e2 = std::get<GCFMExtras>(*s2.extras);
    const Ellipse E1{e1.Av, e1.AMin, e1.BMax, e1.BMin};
    const Ellipse E2{e2.Av, e2.AMin, e2.BMax, e2.BMin};
    const auto v0_1 = agent1.v0.value_or(Defaults::v0);
    const auto v0_2 = s2.v0.value_or(Defaults::v0);
    const double scale1 = (v0_1 == 0.0) ? 1.0 : e1.speed / v0_1;
    const double scale2 = (v0_2 == 0.0) ? 1.0 : e2.speed / v0_2;
    const auto orientation1 = agent1.orientation.value_or(Point{1.0, 0.0});
    const auto orientation2 = s2.orientation.value_or(Point{1.0, 0.0});

    return E1.EffectiveDistanceToEllipse(
        E2,
        agent1.position,
        Pos(agent2),
        scale1,
        scale2,
        e1.speed,
        e2.speed,
        orientation1,
        orientation2);
}
