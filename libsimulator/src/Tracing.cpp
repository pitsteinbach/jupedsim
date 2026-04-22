// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Tracing.hpp"

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(perfetto::Category("main").SetDescription("Main iteration over agents"));

#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <utility>

namespace cr = std::chrono;

// Must be defined exactly once for the track-event namespace to provide
// storage backing symbols like perfetto::...::kCategoryRegistry.
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace
{
perfetto::TraceConfig BuildDefaultTraceConfig()
{
    perfetto::TraceConfig cfg;

    auto* buffer = cfg.add_buffers();
    buffer->set_size_kb(8192);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");

    return cfg;
}
} // namespace

TimerEntry::TimerEntry() : t(0)
{
}

TimerEntry::TimerEntry(TimerEntry&& other) noexcept
    : startedAt(std::move(other.startedAt)), t(other.t), running(other.running)
{
    other.t = 0;
    other.running = false;
}

TimerEntry& TimerEntry::operator=(TimerEntry&& other) noexcept
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

TimerEntry::TimerEntry(const TimerEntry& other)
    : startedAt(other.startedAt), t(other.t), running(other.running)
{
}

TimerEntry& TimerEntry::operator=(const TimerEntry& other)
{
    if(this != &other) {
        startedAt = other.startedAt;
        t = other.t;
        running = other.running;
    }
    return *this;
}

void TimerEntry::start()
{
    if(!running) {
        running = true;
        startedAt = cr::high_resolution_clock::now();
    }
}

void TimerEntry::stop()
{
    if(running) {
        running = false;
        t += std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - startedAt)
                 .count();
    }
}

uint64_t TimerEntry::getDuration() const
{
    return t;
}

PerfStats::~PerfStats()
{
}

void PerfStats::CreateProfilerSession()
{
    if(tracing_session) {
        return;
    }

    if(!perfetto::Tracing::IsInitialized()) {
        perfetto::TracingInitArgs args;
        args.backends |= perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
    }

    perfetto::TrackEvent::Register();

    tracing_session = perfetto::Tracing::NewTrace();
    tracing_session->Setup(BuildDefaultTraceConfig());
    tracing_session->StartBlocking();
}

void PerfStats::DumpProfilerSession(const std::string& filename)
{
    if(!tracing_session) {
        return;
    }

    perfetto::TrackEvent::Flush();
    tracing_session->StopBlocking();
    const auto trace_data = tracing_session->ReadTraceBlocking();

    std::ofstream output(filename, std::ios::binary | std::ios::trunc);
    if(output.is_open()) {
        output.write(trace_data.data(), static_cast<std::streamsize>(trace_data.size()));
        output.flush();
    } else {
        std::cerr << "Failed to open Perfetto output file: " << filename << std::endl;
    }

    tracing_session.reset();
}

void PerfStats::EnableProfiler(bool status)
{
    if(status == enable_tracing) {
        return;
    }

    enable_tracing = status;
}

void PerfStats::PushProfilerProbe(const std::string& name)
{
    if(!tracing_session) {
        CreateProfilerSession();
    }
    TRACE_EVENT_BEGIN("main", perfetto::DynamicString{name.c_str()});
}
void PerfStats::PopProfilerProbe()
{
    TRACE_EVENT_END("main");
}

void PerfStats::PushTimerProbe(const std::string& name, int loglevel)
{
    if(loglevel > log_level) {
        return;
    }
    if(timer_map.find(name) == timer_map.end()) {
        timer_map[name] = TimerEntry();
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
        PopProfilerProbe();
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
