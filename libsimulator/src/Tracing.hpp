// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <perfetto.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

class TimerEntry
{
    std::chrono::high_resolution_clock::time_point startedAt;
    uint64_t t;
    bool running{false};

public:
    TimerEntry();
    ~TimerEntry() {}
    TimerEntry(const TimerEntry& other);
    TimerEntry& operator=(const TimerEntry& other);
    TimerEntry(TimerEntry&& other) noexcept;
    TimerEntry& operator=(TimerEntry&& other) noexcept;
    void start();
    void stop();
    uint64_t getDuration() const;
};

class PerfStats
{
    bool enable_tracing{false};
    int log_level{0};
    std::unordered_map<std::string, TimerEntry> timer_map{};
    std::shared_ptr<perfetto::TracingSession> tracing_session{};

public:
    ~PerfStats();
    void PushTimerProbe(const std::string& name, int loglevel = 0);
    void PopTimerProbe(const std::string& name);
    void EnableProfiler(bool status);
    uint64_t GetTimerEntry(const std::string& name) const;
    void DumpProfilerSession(const std::string& filename);
    void PrintTimerEntries() const;
    std::map<std::string, uint64_t> GetTimerEntries() const
    {
        std::map<std::string, uint64_t> entries;
        for(const auto& [name, trace] : timer_map) {
            entries[name] = trace.getDuration();
        }
        return entries;
    }
    void SetLogLevel(int level) { log_level = level; };
    int GetLogLevel() const { return log_level; };

private:
    void PushProfilerProbe(const std::string& name);
    void PopProfilerProbe();
    void CreateProfilerSession();
};
