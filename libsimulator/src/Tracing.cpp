// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Tracing.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <utility>

#ifdef BUILD_PROFILER
#include "ProfilerLib/DurationEvent.hpp"
#endif

namespace cr = std::chrono;

Trace::Trace() : t(0)
{
}

Trace::Trace(Trace&& other) noexcept
    : startedAt(std::move(other.startedAt)), t(other.t), running(other.running)
{
    other.t = 0;
    other.running = false;
}

Trace& Trace::operator=(Trace&& other) noexcept
{
    if(this != &other) {
        startedAt = std::move(other.startedAt);
        t = other.t;
        running = other.running;
        other.t = 0;
        other.running = false;
    }
    return *this;
}

Trace::Trace(const Trace& other) : startedAt(other.startedAt), t(other.t), running(other.running)
{
}

Trace& Trace::operator=(const Trace& other)
{
    if(this != &other) {
        startedAt = other.startedAt;
        t = other.t;
        running = other.running;
    }
    return *this;
}

void Trace::start()
{
    if(!running) {
        running = true;
        startedAt = cr::high_resolution_clock::now();
    }
}

void Trace::stop()
{
    if(running) {
        running = false;
        t += std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - startedAt)
                 .count();
    }
}

uint64_t Trace::getDuration() const
{
    return t;
}

#ifdef BUILD_PROFILER
void PerfStats::PushProfilerProbe(const std::string& name)
{
    if(!profiler) {
        profiler = std::make_shared<Profiler>("JPS Simulator", prof_filename);
    }
    if(event_map.find(name) == event_map.end()) {
        std::string event_name = "JPS Simulator - " + name;
        event_map[name] = std::make_shared<DurationEvent>(*profiler, event_name);
        event_map[name]->start();
    } else {
        event_map[name]->start();
        // Probe already exists, return
        return;
    }
}
void PerfStats::PopProfilerProbe(const std::string& name)
{
    if(event_map.find(name) != event_map.end()) {
        event_map[name]->stop();
    }
}

#else
void PerfStats::PushProfilerProbe(const std::string& name)
{
}
void PerfStats::PopProfilerProbe(const std::string& name)
{
}
#endif

void PerfStats::PushTimerProbe(const std::string& name, int loglevel)
{
    if (loglevel > log_level) {
        return;
    }
    if(timer_map.find(name) == timer_map.end()) {
        timer_map[name] = Trace();
        timer_map[name].start();
    } else {
        timer_map[name].start();
    }
    if(enable_tracing) {
        PushProfilerProbe(name);
    }
}

void PerfStats::PopTimerProbe(const std::string& name)
{
    if(timer_map.find(name) != timer_map.end()) {
        timer_map[name].stop();
    }
    if(enable_tracing) {
        PopProfilerProbe(name);
    }
}

void PerfStats::PrintTimerEntries() const
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::endl;
    std::cout << "Timer Entries:" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    uint64_t total_iteration_duration = 0;
    total_iteration_duration = timer_map.find("TotalIteration") != timer_map.end() ?
                                   timer_map.at("TotalIteration").getDuration() :
                                   0;
    for(const auto& [name, trace] : timer_map) {
        if(name != "TotalIteration") {
            std::string tabs;
            if(name.length() < 16) {
                tabs = "\t\t\t";
            } else if(name.length() < 24) {
                tabs = "\t\t";
            } else if(name.length() < 50) {
                tabs = "\t";
            }
            float percentage =
                total_iteration_duration > 0 ?
                    (static_cast<float>(trace.getDuration()) / total_iteration_duration) * 100.0f :
                    0.0f;
            std::cout << name << tabs << float(trace.getDuration()) / 1000000 << " s " << "("
                      << std::setprecision(2) << percentage << "%)" << std::endl;
            continue;
        }
    }
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Total Iteration:" << "\t\t" << float(total_iteration_duration) / 1000000 << " s "
              << "(100%)" << std::endl;
}

uint64_t PerfStats::GetTimerEntry(const std::string& name) const
{
    if(timer_map.find(name) != timer_map.end()) {
        return timer_map.at(name).getDuration();
    }
    return 0;
}
