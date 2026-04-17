// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Tracing.hpp"
#include <optional>

#ifdef BUILD_PROFILER
    #include "ProfilerLib/DurationEvent.hpp"
    #include <iostream>
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
    if (profiler == nullptr)
    {
        std::cout << "Initializing profiler with filename: " << prof_filename << std::endl;
        profiler = std::make_unique<Profiler>("JPS Simulator", prof_filename);
        profiler->setProcessName("Flow Example");
        profiler->setThreadName("main thread");
    }
    if (event_map.find(name) == event_map.end()) {
        std::cout << "Creating profiler probe: " << name << std::endl;
        std::string event_name = name;
        event_map[name] = std::make_unique<DurationEvent>(*profiler, event_name);
        event_map[name]->start();
    }
    else
    {
        event_map[name]->start();
        return;
    } 
}
void PerfStats::PopProfilerProbe(const std::string& name)
{
    if (event_map.find(name) != event_map.end()) {
        if (event_map[name])
        {
            event_map[name]->stop();
        }
        
    }
}

    PerfStats::~PerfStats() {
        std::cout << "Finalizing profiler and writing data to file: " << prof_filename <<  std::endl;
        profiler.reset();
    }

#else
void PerfStats::PushProfilerProbe(const std::string& name) {  }
void PerfStats::PopProfilerProbe(const std::string& name) {  }
#endif
