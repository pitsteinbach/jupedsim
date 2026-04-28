// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModelType.hpp"
#include "Tracing.hpp"
#include "ParallelRuntime.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

class OperationalDecisionSystem
{
    std::unique_ptr<OperationalModel> _model{};

public:
    OperationalDecisionSystem(std::unique_ptr<OperationalModel>&& model) : _model(std::move(model))
    {
    }
    ~OperationalDecisionSystem() = default;
    OperationalDecisionSystem(const OperationalDecisionSystem& other) = delete;
    OperationalDecisionSystem& operator=(const OperationalDecisionSystem& other) = delete;
    OperationalDecisionSystem(OperationalDecisionSystem&& other) = delete;
    OperationalDecisionSystem& operator=(OperationalDecisionSystem&& other) = delete;

    OperationalModelType ModelType() const { return _model->Type(); }

    void
    Run(double dT,
        double /*t_in_sec*/,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
        const CollisionGeometry& geometry,
        std::vector<GenericAgent>& agents) const
    {
        std::vector<std::optional<OperationalModelUpdate>> updates{};
        updates.resize(agents.size());
        {
            JPS_SCOPED_PROBE(ProfilerSingleton::instance(), "OperationalDecisionSystem::ComputeUpdates");
#ifdef JPS_USE_STD_PARALLEL_ALGORITHMS
            std::transform(std::execution::par, std::begin(agents), std::end(agents), std::begin(updates), [this, &dT, &geometry, &neighborhoodSearch](const auto& agent) {
                return _model->ComputeNewPosition(dT, agent, geometry, neighborhoodSearch);
            });
#else
            ParallelRuntime::Instance().parallelFor(0, agents.size(),
            [this, &dT, &geometry, &neighborhoodSearch, &agents, &updates](size_t idx) {
                updates[idx] =
                    _model->ComputeNewPosition(dT, agents[idx], geometry, neighborhoodSearch);
            });
#endif
        }

        {
#ifdef JPS_USE_STD_PARALLEL_ALGORITHMS
            JPS_SCOPED_PROBE(ProfilerSingleton::instance(), "OperationalDecisionSystem::ApplyUpdates");
           
        std::for_each(

            boost::make_zip_iterator(boost::make_tuple(std::begin(agents), std::begin(updates))),
            boost::make_zip_iterator(boost::make_tuple(std::end(agents), std::end(updates))),
                       [this](auto tup) {
                auto& [agent, update] = tup;
                if(update) {
                    _model->ApplyUpdate(*update, agent);
                }});
                

#else
            JPS_SCOPED_PROBE(ProfilerSingleton::instance(), "OperationalDecisionSystem::ApplyUpdates"); 
            ParallelRuntime::Instance().parallelFor(0, agents.size(),
            [this, &agents, &updates](size_t idx) {
                auto& agent = agents[idx];
                auto& update = updates[idx];
                if(update) {
                    _model->ApplyUpdate(*update, agent);
                }
            });
#endif
        }
    }

    void ValidateAgent(
        const GenericAgent& agent,
        const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
        const CollisionGeometry& geometry) const
    {
        _model->CheckModelConstraint(agent, neighborhoodSearch, geometry);
    }
};
