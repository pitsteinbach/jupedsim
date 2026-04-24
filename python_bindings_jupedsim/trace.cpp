// SPDX-License-Identifier: LGPL-3.0-or-later

#include "Tracing.hpp"
#include "Timing.hpp"

#include <fmt/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_trace(py::module_& m)
{
    py::class_<ProfilerSingleton>(m, "Trace")
        .def_static(
            "instance",
            []() -> ProfilerSingleton& { return ProfilerSingleton::instance(); },
            py::return_value_policy::reference)
        .def("enable", [](ProfilerSingleton& ps) { ps.enable(); })
        .def("disable", [](ProfilerSingleton& ps) { ps.disable(); })
        .def("push_probe", [](ProfilerSingleton& ps, const std::string& name) { ps.pushProbe(name); })
        .def("pop_probe",[](ProfilerSingleton& ps) { ps.popProbe(); })
        .def("dump", [](ProfilerSingleton& ps, const std::string& filename) { ps.dump(filename); });

    py::class_<Timer>(m, "Timer")
        .def("get_timer_entry", [](const Timer& t, const std::string& name) { return t.getTimerEntry(name); })
        .def("get_timer_entries", [](const Timer& t) { return t.getTimerEntries(); })
        .def("push_timer", [](Timer& t, const std::string& name) { t.pushTimerProbe(name); })
        .def("pop_timer", [](Timer& t, const std::string& name) { t.popTimerProbe(name); });

}