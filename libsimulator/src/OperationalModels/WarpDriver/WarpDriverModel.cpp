// SPDX-License-Identifier: LGPL-3.0-or-later
//
// WarpDriver collision-avoidance model.
// Based on Wolinski, Lin, and Pettré (2016) — PhD thesis Chapter 4, Appendix B.
//
// Warp operators implemented (B.1–B.15):
//   W_ref  — local frame change (W_local: A's frame → B's frame)
//   W_th   — time horizon normalization (B.1–B.3)
//   W_tu   — time uncertainty with probability scaling (B.4–B.6)
//   W_r    — radius normalization via Minkowski sum (B.7–B.9)
//   W_v    — velocity shear (B.10–B.12)
//   W_vu   — anisotropic velocity uncertainty with probability scaling (B.13–B.15)
//
// Non-thesis safety mechanisms (practical additions):
//   - Short-range repulsion (3× combined radius) to prevent overlaps
//   - Boundary wall steering
//   - Stuck detection with lateral detour to break narrow-passage deadlocks
//   - Lateral perturbation for symmetry breaking
//
// TODO: W_ref currently uses simple W_local (straight-line frame change).
//   The thesis defines graph-based variants (Algorithm 3) that warp space
//   along navigable paths — W_el (environment layout), W_io (obstacle
//   interactions), W_ob (observed behaviors). These would enable anticipatory
//   avoidance around corners and bends. The routing infrastructure exists
//   (RoutingEngine::ComputeAllWaypoints provides the full waypoint path);
//   the path could serve as the graph for Algorithm 3's spatial projection.
//
#include "WarpDriverModel.hpp"

#include "AgentState.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModelType.hpp"
#include "Point.hpp"
#include "SimulationError.hpp"

#include <algorithm>
#include <cmath>
#include <variant>

// ============================================================================
// IntrinsicField
// ============================================================================

void WarpDriverModel::IntrinsicField::Compute(double sigma)
{
    nx = static_cast<int>(std::round((xMax - xMin) / dx)) + 1;
    ny = static_cast<int>(std::round((yMax - yMin) / dy)) + 1;
    values.resize(static_cast<size_t>(nx * ny), 0.0);
    gradients.resize(static_cast<size_t>(nx * ny), Point{0.0, 0.0});

    const double sigma_squared = sigma * sigma;

    const double integrationStep = 0.05;
    const double integrationRadius = 1.0;

    for(int ix = 0; ix < nx; ++ix) {
        for(int iy = 0; iy < ny; ++iy) {
            const double px = xMin + ix * dx;
            const double py = yMin + iy * dy;

            double val = 0.0;
            for(double u = -integrationRadius; u <= integrationRadius; u += integrationStep) {
                for(double v = -integrationRadius; v <= integrationRadius; v += integrationStep) {
                    if(u * u + v * v <= 1.0) {
                        double dx2 = px - u;
                        double dy2 = py - v;
                        val += std::exp(-(dx2 * dx2 + dy2 * dy2) / (2.0 * sigma_squared));
                    }
                }
            }
            val *= integrationStep * integrationStep;
            values[static_cast<size_t>(ix * ny + iy)] = val;
        }
    }

    const double maxVal = *std::max_element(values.begin(), values.end());
    if(maxVal > 0.0) {
        for(auto& v : values) {
            v /= maxVal;
        }
    }

    for(int ix = 0; ix < nx; ++ix) {
        for(int iy = 0; iy < ny; ++iy) {
            double dIdx = 0.0;
            double dIdy = 0.0;
            if(ix > 0 && ix < nx - 1) {
                dIdx = (values[static_cast<size_t>((ix + 1) * ny + iy)] -
                        values[static_cast<size_t>((ix - 1) * ny + iy)]) /
                       (2.0 * dx);
            }
            if(iy > 0 && iy < ny - 1) {
                dIdy = (values[static_cast<size_t>(ix * ny + (iy + 1))] -
                        values[static_cast<size_t>(ix * ny + (iy - 1))]) /
                       (2.0 * dy);
            }
            gradients[static_cast<size_t>(ix * ny + iy)] = Point{dIdx, dIdy};
        }
    }
}

std::pair<double, Point> WarpDriverModel::IntrinsicField::Sample(double x, double y) const
{
    if(x < xMin || x > xMax || y < yMin || y > yMax) {
        return {0.0, Point{0.0, 0.0}};
    }

    const double fx = (x - xMin) / dx;
    const double fy = (y - yMin) / dy;
    const int ix = std::clamp(static_cast<int>(fx), 0, nx - 2);
    const int iy = std::clamp(static_cast<int>(fy), 0, ny - 2);
    const double sx = fx - ix;
    const double sy = fy - iy;

    const auto idx = [&](int i, int j) -> size_t { return static_cast<size_t>(i * ny + j); };

    const double v00 = values[idx(ix, iy)];
    const double v10 = values[idx(ix + 1, iy)];
    const double v01 = values[idx(ix, iy + 1)];
    const double v11 = values[idx(ix + 1, iy + 1)];
    const double val =
        v00 * (1 - sx) * (1 - sy) + v10 * sx * (1 - sy) + v01 * (1 - sx) * sy + v11 * sx * sy;

    const Point g00 = gradients[idx(ix, iy)];
    const Point g10 = gradients[idx(ix + 1, iy)];
    const Point g01 = gradients[idx(ix, iy + 1)];
    const Point g11 = gradients[idx(ix + 1, iy + 1)];
    const Point grad = g00 * ((1 - sx) * (1 - sy)) + g10 * (sx * (1 - sy)) +
                       g01 * ((1 - sx) * sy) + g11 * (sx * sy);

    return {val, grad};
}

// ============================================================================
// Warp Operators
// ============================================================================

namespace
{

using STP = WarpDriverModel::SpaceTimePoint;

STP WarpLocalForward(const STP& s, Point posA, Point orientA, Point posB, Point orientB)
{
    const double cosA = orientA.x;
    const double sinA = orientA.y;
    const double wx = cosA * s.x - sinA * s.y + posA.x;
    const double wy = sinA * s.x + cosA * s.y + posA.y;

    const double dx = wx - posB.x;
    const double dy = wy - posB.y;
    const double cosB = orientB.x;
    const double sinB = orientB.y;
    return STP{cosB * dx + sinB * dy, -sinB * dx + cosB * dy, s.t};
}

STP WarpVelocityForward(const STP& s, double speedB)
{
    return STP{s.x - speedB * s.t, s.y, s.t};
}

STP WarpRadiusForward(const STP& s, double radiusB)
{
    const double invR = 1.0 / std::max(radiusB, 1e-6);
    return STP{s.x * invR, s.y * invR, s.t};
}

STP WarpTimeUncertaintyForward(const STP& s, double lambda)
{
    const double scale = 1.0 / (1.0 + lambda * std::max(s.t, 0.0));
    return STP{s.x * scale, s.y * scale, s.t};
}

struct VelocityUncertaintyScale {
    double beta1;
    double beta2;
};

VelocityUncertaintyScale VelocityUncertaintyFactors(double uncertaintyX, double uncertaintyY)
{
    return {1.0 / (1.0 + uncertaintyX), 1.0 + uncertaintyY};
}

STP WarpVelocityUncertaintyForward(const STP& s, double uncertaintyX, double uncertaintyY)
{
    const auto [beta1, beta2] = VelocityUncertaintyFactors(uncertaintyX, uncertaintyY);
    return STP{s.x * beta1, s.y * beta2, s.t};
}

struct WarpParams {
    Point posA;
    Point orientA;
    Point posB;
    Point orientB;
    double speedB;
    double radiusB;
    double lambda;
    double velocityUncertaintyX;
    double velocityUncertaintyY;
    double timeHorizon;
};

double ProbabilityScale(const STP& sOriginal, const WarpParams& p)
{
    const double beta_tu = 1.0 / (1.0 + p.lambda * std::max(sOriginal.t, 0.0));
    const auto [beta1, beta2] =
        VelocityUncertaintyFactors(p.velocityUncertaintyX, p.velocityUncertaintyY);
    return beta_tu * beta_tu * beta1 * beta2;
}

STP ComposeForward(const STP& s, const WarpParams& p)
{
    auto s1 = WarpLocalForward(s, p.posA, p.orientA, p.posB, p.orientB);
    auto s2 = WarpVelocityForward(s1, p.speedB);
    auto s3 = WarpRadiusForward(s2, p.radiusB);
    auto s4 = WarpTimeUncertaintyForward(s3, p.lambda);
    auto s5 = WarpVelocityUncertaintyForward(s4, p.velocityUncertaintyX, p.velocityUncertaintyY);
    s5.t = (p.timeHorizon > 0.0) ? s5.t / p.timeHorizon : 0.0;
    return s5;
}

STP ComposeGradientInverse(const Point& gradI, const STP& sOriginal, const WarpParams& p)
{
    double gx = gradI.x;
    double gy = gradI.y;
    double gt = 0.0;

    {
        const auto [beta1, beta2] =
            VelocityUncertaintyFactors(p.velocityUncertaintyX, p.velocityUncertaintyY);
        gx *= beta1;
        gy *= beta2;
    }

    {
        const double t = sOriginal.t;
        const double beta = 1.0 / (1.0 + p.lambda * std::max(t, 0.0));
        auto sAtTu = WarpLocalForward(sOriginal, p.posA, p.orientA, p.posB, p.orientB);
        sAtTu = WarpVelocityForward(sAtTu, p.speedB);
        sAtTu = WarpRadiusForward(sAtTu, p.radiusB);
        const double gamma1 = -p.lambda * beta * beta * sAtTu.x;
        const double gamma2 = -p.lambda * beta * beta * sAtTu.y;
        const double gxOld = gx;
        const double gyOld = gy;
        gx *= beta;
        gy *= beta;
        gt = gamma1 * gxOld + gamma2 * gyOld + gt;
    }

    {
        gt -= p.speedB * gx;
    }

    {
        const double cosA = p.orientA.x;
        const double sinA = p.orientA.y;
        const double cosB = p.orientB.x;
        const double sinB = p.orientB.y;

        const double cos_ab = cosA * cosB + sinA * sinB;
        const double sin_ab = sinA * cosB - cosA * sinB;
        const double gx_new = cos_ab * gx + sin_ab * gy;
        const double gy_new = -sin_ab * gx + cos_ab * gy;
        gx = gx_new;
        gy = gy_new;
    }

    return STP{gx, gy, gt};
}

} // anonymous namespace

// ============================================================================
// WarpDriverModel
// ============================================================================

WarpDriverModel::WarpDriverModel(double sigma, uint64_t rngSeed)
    : _cutOffRadius(2.0 * 1.5 * 2.0 + 2.0 * 0.3 + 0.5), _rng(rngSeed)
{
    if(sigma <= 0.0) {
        throw SimulationError("WarpDriverModel: sigma must be > 0, got {}", sigma);
    }
    _intrinsicField.Compute(sigma);
}

AgentState WarpDriverModel::MakeState(Point pos)
{
    return AgentState{
        .type = OperationalModelType::WARP_DRIVER,
        .position = pos,
        .orientation = Point{0.0, 0.0},
        .v0 = Defaults::v0,
        .radius = Defaults::radius,
        .extras = WarpExtras{
            .stuckTime = 0.0,
            .anchorX = 0.0,
            .anchorY = 0.0,
            .detourTime = 0.0,
            .detourSide = 1,
            .timeHorizon = Defaults::timeHorizon,
            .stepSize = Defaults::stepSize,
            .timeUncertainty = Defaults::timeUncertainty,
            .velocityUncertaintyX = Defaults::velocityUncertaintyX,
            .velocityUncertaintyY = Defaults::velocityUncertaintyY,
            .numSamples = Defaults::numSamples,
        },
    };
}

OperationalModelType WarpDriverModel::Type() const
{
    return OperationalModelType::WARP_DRIVER;
}

void WarpDriverModel::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& /*neighborhoodSearch*/,
    const CollisionGeometry& /*geometry*/) const
{
    const auto* statePtr = std::get_if<AgentState>(&agent.model);
    if(!statePtr) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} does not have WarpDriverModel data",
            agent.id);
    }
    const auto& state = *statePtr;
    const auto* extrasPtr = std::get_if<WarpExtras>(&state.extras);
    if(!extrasPtr) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} missing WarpExtras", agent.id);
    }
    const auto& extras = *extrasPtr;
    const auto radius = state.radius.value_or(Defaults::radius);
    const auto v0 = state.v0.value_or(Defaults::v0);

    if(radius <= 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid radius {}", agent.id, radius);
    }
    if(v0 < 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid v0 {}", agent.id, v0);
    }
    if(extras.timeHorizon <= 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid timeHorizon {}, must be > 0",
            agent.id,
            extras.timeHorizon);
    }
    if(extras.stepSize <= 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid stepSize {}, must be > 0",
            agent.id,
            extras.stepSize);
    }
    if(extras.numSamples < 1) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid numSamples {}, must be >= 1",
            agent.id,
            extras.numSamples);
    }
    if(extras.timeUncertainty < 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid timeUncertainty {}, must be "
            ">= 0",
            agent.id,
            extras.timeUncertainty);
    }
    if(extras.velocityUncertaintyX < 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid velocityUncertaintyX {}, must "
            "be >= 0",
            agent.id,
            extras.velocityUncertaintyX);
    }
    if(extras.velocityUncertaintyY < 0.0) {
        throw SimulationError(
            "WarpDriverModel constraint check: agent {} has invalid velocityUncertaintyY {}, must "
            "be >= 0",
            agent.id,
            extras.velocityUncertaintyY);
    }
}

void WarpDriverModel::ComputeNext(
    double dT,
    const GenericAgent& current,
    GenericAgent& next,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    const auto& state = std::get<AgentState>(current.model);
    const auto& extras = std::get<WarpExtras>(state.extras);
    const double speed = state.v0.value_or(Defaults::v0);
    const double agentRadius = state.radius.value_or(Defaults::radius);

    Point orient = state.orientation.value_or(Point{0.0, 0.0});
    if(orient.Norm() < 1e-9) {
        orient = Point{1.0, 0.0};
    } else {
        orient = orient.Normalized();
    }

    Point toTarget = current.destination - Pos(current);
    const double distToTarget = toTarget.Norm();
    if(distToTarget < 1e-9) {
        auto& nextState = std::get<AgentState>(next.model);
        auto& nextExtras = std::get<WarpExtras>(nextState.extras);
        nextState.position = Pos(current);
        nextState.orientation = orient;
        nextExtras.stuckTime = 0.0;
        nextExtras.anchorX = 0.0;
        nextExtras.anchorY = 0.0;
        nextExtras.detourTime = 0.0;
        nextExtras.detourSide = 1;
        return;
    }
    Point desiredDir = toTarget.Normalized();
    Point effectiveOrient = desiredDir;

    const double dtSample = extras.timeHorizon / std::max(extras.numSamples - 1, 1);
    const auto neighbors = neighborhoodSearch.GetNeighboringAgents(Pos(current), _cutOffRadius);

    Point repulsion{0.0, 0.0};
    for(const auto& neighbor : neighbors) {
        if(neighbor.id == current.id) {
            continue;
        }
        const auto* nbState = std::get_if<AgentState>(&neighbor.model);
        if(!nbState) {
            continue;
        }
        const double nbRadius = nbState->radius.value_or(Defaults::radius);
        Point diff = Pos(current) - Pos(neighbor);
        const double dist = diff.Norm();
        const double combinedRadius = agentRadius + nbRadius;
        if(dist < combinedRadius * 3.0 && dist > 1e-6) {
            const double overlap = combinedRadius * 3.0 - dist;
            repulsion = repulsion + diff.Normalized() * (speed * overlap / dist);
        } else if(dist <= 1e-6) {
            repulsion = repulsion + Point{-desiredDir.y, desiredDir.x} * speed;
        }
    }

    std::uniform_real_distribution<double> perturbDist(-0.05, 0.05);

    struct Sample {
        double t;
        STP r;
        double pTotal;
        STP gradTotal;
    };
    std::vector<Sample> samples(static_cast<size_t>(extras.numSamples));

    for(int i = 0; i < extras.numSamples; ++i) {
        const double t = i * dtSample;
        const double lateralPerturbation = perturbDist(_rng);
        samples[static_cast<size_t>(i)] =
            Sample{t, STP{speed * t, lateralPerturbation, t}, 0.0, STP{0, 0, 0}};
    }

    for(const auto& neighbor : neighbors) {
        if(neighbor.id == current.id) {
            continue;
        }
        const auto* nbState = std::get_if<AgentState>(&neighbor.model);
        if(!nbState) {
            continue;
        }

        Point nbOrient = nbState->orientation.value_or(Point{0.0, 0.0});
        if(nbOrient.Norm() < 1e-9) {
            nbOrient = Point{1.0, 0.0};
        } else {
            nbOrient = nbOrient.Normalized();
        }
        const double nbSpeed = nbState->v0.value_or(Defaults::v0);
        const double nbRadius = nbState->radius.value_or(Defaults::radius);

        WarpParams wp{};
        wp.posA = Pos(current);
        wp.orientA = effectiveOrient;
        wp.posB = Pos(neighbor);
        wp.orientB = nbOrient;
        wp.speedB = nbSpeed;
        wp.radiusB = agentRadius + nbRadius;
        wp.lambda = extras.timeUncertainty;
        wp.velocityUncertaintyX = extras.velocityUncertaintyX;
        wp.velocityUncertaintyY = extras.velocityUncertaintyY;
        wp.timeHorizon = extras.timeHorizon;

        for(auto& s : samples) {
            STP warped = ComposeForward(s.r, wp);

            if(warped.t < 0.0 || warped.t > 1.0) {
                continue;
            }

            auto [intrinsicP, gradI] = _intrinsicField.Sample(warped.x, warped.y);
            const double pB = intrinsicP * ProbabilityScale(s.r, wp);

            if(pB < 1e-12) {
                continue;
            }

            STP gradB = ComposeGradientInverse(gradI, s.r, wp);

            double pOld = s.pTotal;
            s.pTotal = pOld + pB - pOld * pB;
            s.gradTotal.x = s.gradTotal.x + gradB.x - pOld * gradB.x - pB * s.gradTotal.x;
            s.gradTotal.y = s.gradTotal.y + gradB.y - pOld * gradB.y - pB * s.gradTotal.y;
            s.gradTotal.t = s.gradTotal.t + gradB.t - pOld * gradB.t - pB * s.gradTotal.t;
        }
    }

    double N = 0.0;
    double P = 0.0;
    STP G{0, 0, 0};
    STP S{0, 0, 0};

    for(const auto& s : samples) {
        N += s.pTotal * dtSample;
        P += s.pTotal * s.pTotal * dtSample;
        G.x += s.pTotal * s.gradTotal.x * dtSample;
        G.y += s.pTotal * s.gradTotal.y * dtSample;
        G.t += s.pTotal * s.gradTotal.t * dtSample;
        S.x += s.pTotal * s.r.x * dtSample;
        S.y += s.pTotal * s.r.y * dtSample;
        S.t += s.pTotal * s.r.t * dtSample;
    }

    Point newVelLocal;

    if(N < 1e-9) {
        newVelLocal = Point{speed, 0.0};
    } else {
        P /= N;
        G.x /= N;
        G.y /= N;
        G.t /= N;
        S.x /= N;
        S.y /= N;
        S.t /= N;

        STP q{};
        q.x = S.x - extras.stepSize * P * G.x;
        q.y = S.y - extras.stepSize * P * G.y;
        q.t = S.t - extras.stepSize * P * G.t;

        if(q.t > 1e-9) {
            newVelLocal = Point{q.x / q.t, q.y / q.t};
        } else {
            newVelLocal = Point{speed, 0.0};
        }
    }

    const double newSpeed = std::min(newVelLocal.Norm(), speed);

    Point newVelWorld;
    if(newSpeed > 1e-9) {
        Point newDirLocal = newVelLocal.Normalized();
        newVelWorld =
            Point{
                effectiveOrient.x * newDirLocal.x - effectiveOrient.y * newDirLocal.y,
                effectiveOrient.y * newDirLocal.x + effectiveOrient.x * newDirLocal.y} *
            newSpeed;
    } else {
        newVelWorld = desiredDir * speed * 0.01;
    }

    newVelWorld = newVelWorld + repulsion;

    const auto& walls = geometry.LineSegmentsInApproxDistanceTo(Pos(current));
    for(const auto& wall : walls) {
        const Point wallVec = wall.p2 - wall.p1;
        const double wallLen2 = wallVec.ScalarProduct(wallVec);
        if(wallLen2 < 1e-12) {
            continue;
        }
        const Point toAgent = Pos(current) - wall.p1;
        const double t = std::clamp(toAgent.ScalarProduct(wallVec) / wallLen2, 0.0, 1.0);
        const Point closest = wall.p1 + wallVec * t;
        const Point diff = Pos(current) - closest;
        const double dist = diff.Norm();
        if(dist < agentRadius * 3.0 && dist > 1e-6) {
            const double steering = speed * (agentRadius * 3.0 - dist) / dist;
            newVelWorld = newVelWorld + diff.Normalized() * steering;
        }
    }

    double finalSpeed = newVelWorld.Norm();
    if(finalSpeed > speed && finalSpeed > 1e-9) {
        newVelWorld = newVelWorld * (speed / finalSpeed);
        finalSpeed = speed;
    }

    double stuckTime = extras.stuckTime;
    double anchorX = extras.anchorX;
    double anchorY = extras.anchorY;
    double detourTime = extras.detourTime;
    int detourSide = extras.detourSide;

    auto& nextState = std::get<AgentState>(next.model);
    auto& nextExtras = std::get<WarpExtras>(nextState.extras);

    if(detourTime > 0.0) {
        detourTime -= dT;
        Point lateral{-desiredDir.y * detourSide, desiredDir.x * detourSide};
        Point detourDir = (lateral * 0.8 + desiredDir * 0.2).Normalized();
        Point detourVel = detourDir * speed * 0.5;
        Point newPos = Pos(current) + detourVel * dT;
        if(!geometry.InsideGeometry(newPos)) {
            detourSide = -detourSide;
            lateral = Point{-desiredDir.y * detourSide, desiredDir.x * detourSide};
            detourDir = (lateral * 0.8 + desiredDir * 0.2).Normalized();
            detourVel = detourDir * speed * 0.5;
            newPos = Pos(current) + detourVel * dT;
            if(!geometry.InsideGeometry(newPos)) {
                newPos = Pos(current) + desiredDir * speed * 0.1 * dT;
                detourDir = desiredDir;
            }
        }
        if(detourTime <= 0.0) {
            detourTime = 0.0;
            stuckTime = 0.0;
            anchorX = newPos.x;
            anchorY = newPos.y;
        }
        nextState.position = newPos;
        nextState.orientation = detourDir;
        nextExtras.stuckTime = stuckTime;
        nextExtras.anchorX = anchorX;
        nextExtras.anchorY = anchorY;
        nextExtras.detourTime = detourTime;
        nextExtras.detourSide = detourSide;
        return;
    }

    constexpr double stuckThreshold = 5.0;
    constexpr double detourDuration = 1.0;
    constexpr double progressRadius = 0.3;

    stuckTime += dT;
    const double netDisplacement = std::hypot(Pos(current).x - anchorX, Pos(current).y - anchorY);

    if(netDisplacement > progressRadius) {
        stuckTime = 0.0;
        anchorX = Pos(current).x;
        anchorY = Pos(current).y;
    } else if(stuckTime >= stuckThreshold) {
        std::uniform_int_distribution<int> sideDist(0, 1);
        detourSide = sideDist(_rng) * 2 - 1;
        detourTime = detourDuration;
        stuckTime = 0.0;
    }

    const double smoothing = 0.5;
    Point smoothedVel = newVelWorld * smoothing + orient * (newVelWorld.Norm() * (1.0 - smoothing));
    double smoothedSpeed = smoothedVel.Norm();
    if(smoothedSpeed > speed && smoothedSpeed > 1e-9) {
        smoothedVel = smoothedVel * (speed / smoothedSpeed);
    }

    Point newPos = Pos(current) + smoothedVel * dT;
    Point newOrient = (smoothedVel.Norm() > 1e-9) ? smoothedVel.Normalized() : orient;

    nextState.position = newPos;
    nextState.orientation = newOrient;
    nextExtras.stuckTime = stuckTime;
    nextExtras.anchorX = anchorX;
    nextExtras.anchorY = anchorY;
    nextExtras.detourTime = detourTime;
    nextExtras.detourSide = detourSide;
}
