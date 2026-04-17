// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <chrono>
#include <optional>
#include <iostream>

    #include <map>
    #include "ProfilerLib/Profiler.hpp"
    #include "ProfilerLib/DurationEvent.hpp"
    


class Trace
{
    std::chrono::high_resolution_clock::time_point startedAt;
    uint64_t& t;

public:
    Trace(uint64_t& _t);
    ~Trace()
    {
        const auto now = std::chrono::high_resolution_clock::now();
        t = std::chrono::duration_cast<std::chrono::microseconds>(now - startedAt).count();
    }
    Trace(const Trace& other) = delete;
    Trace& operator=(const Trace& other) = delete;
    Trace(Trace&& other) = delete;
    Trace& operator=(const Trace&& other) = delete;

};

class PerfStats
{
    uint64_t iterate_duration{};
    uint64_t op_dec_system_run_duration{};
    bool enabled{false};

    std::string prof_filename{"jps_simulator.json"};
    std::unique_ptr<Profiler> profiler;
    std::unique_ptr<DurationEvent> event;
    std::map<std::string, std::unique_ptr<DurationEvent>> event_map{};

public:
    std::optional<Trace> TraceIterate();
    std::optional<Trace> TraceOperationalDecisionSystemRun();
    void PushProfilerProbe(const std::string& name);
    void PopProfilerProbe(const std::string& name);
    void SetEnabled(bool status) { enabled = status; };
    uint64_t IterationDuration() const { return iterate_duration; };
    uint64_t OpDecSystemRunDuration() const { return op_dec_system_run_duration; };
    
    //void SetProfilerFilename(const std::string& filename);
    ~PerfStats();
    
private:
    std::optional<Trace> trace(uint64_t& v);
};
