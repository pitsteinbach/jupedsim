// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Timing.hpp"

#include <fmt/core.h>
#include <perfetto.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <utility>

namespace cr = std::chrono;

TimerEntry::TimerEntry(TimerEntry&& other) noexcept
    : started_at(std::move(other.started_at))
    , duration_in_microseconds(other.duration_in_microseconds)
    , running(other.running)
{
    other.duration_in_microseconds = 0;
    other.running = false;
}

TimerEntry& TimerEntry::operator=(TimerEntry&& other) noexcept
{
    if(this != &other) {
        started_at = std::move(other.started_at);
        duration_in_microseconds = other.duration_in_microseconds;
        running = other.running;
        other.duration_in_microseconds = 0;
        other.running = false;
    }
    return *this;
}

void TimerEntry::start()
{
    if(!running) {
        running = true;
        started_at = cr::high_resolution_clock::now();
    }
}

void TimerEntry::stop()
{
    if(running) {
        running = false;
        duration_in_microseconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::high_resolution_clock::now() - started_at)
                                        .count();
    }
}

uint64_t TimerEntry::getDurationInMicroseconds() const
{
    if(running) {
        return duration_in_microseconds +
               std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - started_at)
                   .count();
    }
    return duration_in_microseconds;
}

void Timer::pushTimerProbe(std::string_view name, int timer_probe_level)
{
    if(timer_probe_level > max_log_level) {
        return;
    }
    std::string name_str(name);
    auto iter = timer_map.find(name_str);
    // use emplace to avoid multiple lookups and unnecessary default construction of TimerEntry
    if(iter == timer_map.end()) {
        timer_map.emplace(name_str, TimerEntry()).first->second.start();
    } else {
        iter->second.start();
    }
    // If tracing is enabled a trace with the same name will also be pushed to the ProfilerSingleton
    // instance.
    if(ProfilerSingleton::instance().isEnabled()) {
        ProfilerSingleton::instance().pushProbe(name);
    }
}

void Timer::popTimerProbe(const std::string_view name)
{
    // use auto iter = timer_map.find(name) to avoid multiple lookups
    auto iter = timer_map.find(std::string(name));
    if(iter != timer_map.end()) {
        iter->second.stop();
    }
    // if tracing is enabled the last entry form the stack will also be popped from the
    // ProfilerSingleton instance.
    if(ProfilerSingleton::instance().isEnabled()) {
        ProfilerSingleton::instance().popProbe();
    }
}

std::string Timer::formatTimerEntries() const
{
    std::string message;
    message += "\n";
    message += "Timer Entries:\n";
    message += "-----------------------------------------------------\n";

    const auto total_iter = timer_map.find("Total Iteration");
    const uint64_t total_iteration_duration =
        (total_iter != timer_map.end()) ? total_iter->second.getDurationInMicroseconds() : 0;

    for(const auto& [name, trace] : timer_map) {
        if(name != "Total Iteration") {
            const float percentage = total_iteration_duration > 0 ?
                                         (static_cast<float>(trace.getDurationInMicroseconds()) /
                                          total_iteration_duration) *
                                             100.0f :
                                         0.0f;
            const auto duration_seconds =
                static_cast<double>(trace.getDurationInMicroseconds()) / 1000000.0;
            message +=
                fmt::format("{:<30}|{:8.2f} s ({:5.2f}%)\n", name, duration_seconds, percentage);
            continue;
        }
    }

    const auto total_seconds = static_cast<double>(total_iteration_duration) / 1000000.0;
    message += "-----------------------------------------------------\n";
    message += fmt::format("{:<30}|{:8.2f} s (100.00%)\n", "Total Simulation Time", total_seconds);
    return message;
}

uint64_t Timer::getDuration(const std::string_view name) const
{
    auto iter = timer_map.find(std::string(name));
    if(iter != timer_map.end()) {
        return iter->second.getDurationInMicroseconds();
    }
    return 0;
}
