// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PROFILER
#include "ProfilerLib/DurationEvent.hpp"
#include "ProfilerLib/Profiler.hpp"
#endif

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>

class Trace
{
    std::chrono::high_resolution_clock::time_point startedAt;
    uint64_t t;
    bool running{false};

public:
    Trace();
    ~Trace() {}
    Trace(const Trace& other);
    Trace& operator=(const Trace& other);
    Trace(Trace&& other) noexcept;
    Trace& operator=(Trace&& other) noexcept;
    void start();
    void stop();
    uint64_t getDuration() const;
};

class PerfStats
{
    mutable uint64_t last_iteration_duration{0};
    mutable uint64_t last_operational_duration{0};
    bool enable_tracing{false};
#ifdef BUILD_PROFILER    
    std::string prof_filename{"jps_simulator.json"};
    std::shared_ptr<Profiler> profiler{};
    std::map<std::string, std::shared_ptr<DurationEvent>> event_map{};
#endif

    std::map<std::string, Trace> timer_map{};

public:
    std::optional<Trace> TraceIterate();
    std::optional<Trace> TraceOperationalDecisionSystemRun();
    void PushTimerProbe(const std::string& name);
    void PopTimerProbe(const std::string& name);
    void EnableProfiler(bool status) { enable_tracing = status; };
    uint64_t IterationDuration() const;
    uint64_t OpDecSystemRunDuration() const;
    void SetProfilerFilename(const std::string& filename) { prof_filename = filename; };
    void PrintTimerEntries() const;
    std::map<std::string, uint64_t> GetTimerEntries() const
    {
        std::map<std::string, uint64_t> entries;
        for(const auto& [name, trace] : timer_map) {
            entries[name] = trace.getDuration();
        }
        return entries;
    }

private:
    void PushProfilerProbe(const std::string& name);
    void PopProfilerProbe(const std::string& name);
};
