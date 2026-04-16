// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Tracing.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <utility>

#ifdef BUILD_PROFILER
    #include "ProfilerLib/DurationEvent.hpp"
#endif

namespace cr = std::chrono;

Trace::Trace(uint64_t& _t) : startedAt(cr::high_resolution_clock::now()), t(_t)
{
}

std::optional<Trace> PerfStats::trace(uint64_t& v)
{
    if(enabled) {
        return std::optional<Trace>{std::in_place, v};
    } else {
        return std::nullopt;
    }
}
std::optional<Trace> PerfStats::TraceIterate()
{
    return trace(iterate_duration);
}

std::optional<Trace> PerfStats::TraceOperationalDecisionSystemRun()
{
    return trace(op_dec_system_run_duration);
}

#ifdef BUILD_PROFILER
void PerfStats::PushProfilerProbe(const std::string& name)
{
    if (event_map.find(name) == event_map.end()) {
        std::string event_name = "JPS Simulator - " + name;
        event_map[name] = std::make_unique<DurationEvent>(profiler, event_name);
        event_map[name]->start();
    }
    else
    {
        // Probe already exists, return
        return;
    } 
}
void PerfStats::PopProfilerProbe(const std::string& name)
{
    if (event_map.find(name) != event_map.end()) {
        event_map[name]->stop();
    }
}

#else
void PerfStats::PushProfilerProbe(const std::string& name) {  }
void PerfStats::PopProfilerProbe(const std::string& name) {  }
#endif
