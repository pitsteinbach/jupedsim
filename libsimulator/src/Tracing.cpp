#include "Tracing.hpp"

#include <perfetto.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace
{
perfetto::TraceConfig buildDefaultTraceConfig()
{
    perfetto::TraceConfig cfg;

    auto* buffer = cfg.add_buffers();
    buffer->set_size_kb(8192);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");

    return cfg;
}
} // namespace

void ProfilerSingleton::createSession()
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
    tracing_session->Setup(buildDefaultTraceConfig());
    tracing_session->StartBlocking();
}

void ProfilerSingleton::writeAndResetSession(const std::string& filename)
{
    if(!tracing_session) {
        return;
    }

    perfetto::TrackEvent::Flush();
    tracing_session->StopBlocking();
    const auto trace_data = tracing_session->ReadTraceBlocking();
    if(filename.empty()) {
        tracing_session.reset();
        return;
    }
    std::ofstream output(filename, std::ios::binary | std::ios::trunc);
    if(output.is_open()) {
        output.write(trace_data.data(), static_cast<std::streamsize>(trace_data.size()));
        output.flush();
    } else {
        std::cerr << "Failed to open Perfetto output file: " << filename << std::endl;
    }

    tracing_session.reset();
}

void ProfilerSingleton::enable()
{
    if(enabled) {
        return;
    }

    createSession();
    enabled = true;
}

void ProfilerSingleton::disable()
{
    if(!enabled && !tracing_session) {
        return;
    }

    writeAndResetSession("");
    enabled = false;
}

void ProfilerSingleton::pushProbe(const std::string& name)
{
    if(!enabled) {
        return;
    }

    TRACE_EVENT_BEGIN("main", perfetto::DynamicString{name.c_str()});
}

void ProfilerSingleton::popProbe()
{
    if(!enabled) {
        return;
    }

    TRACE_EVENT_END("main");
}

void ProfilerSingleton::dump(const std::string& filename)
{
    writeAndResetSession(filename);
    enabled = false;
}

ProfilerSingleton ProfilerSingleton::profiler{};
