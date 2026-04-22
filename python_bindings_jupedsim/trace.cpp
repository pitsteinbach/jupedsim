// SPDX-License-Identifier: LGPL-3.0-or-later

#include "Tracing.hpp"

#include <fmt/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_trace(py::module_& m)
{
    py::class_<PerfStats>(m, "Trace")
        .def("enable_profiler",[](PerfStats& ps, const bool status) { return ps.EnableProfiler(status); })
        .def("get_timer_entry", [](const PerfStats& ps, const std::string& name) { return ps.GetTimerEntry(name); })
        .def("get_timer_entries", [](const PerfStats& ps) { return ps.GetTimerEntries(); })
        .def("push_timer", [](PerfStats& ps, const std::string& name) { ps.PushTimerProbe(name); })
        .def("pop_timer", [](PerfStats& ps, const std::string& name) { ps.PopTimerProbe(name); });
}
