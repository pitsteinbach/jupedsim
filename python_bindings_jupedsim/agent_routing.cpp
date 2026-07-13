// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AgentRouting.hpp"
#include "conversion.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_agent_routing(py::module_& m)
{
    py::class_<AgentRouting>(m, "AgentRouting")
        .def_property_readonly(
            "destination", [](const AgentRouting& r) { return intoTuple(r.destination); })
        .def_property_readonly(
            "target", [](const AgentRouting& r) { return intoTuple(r.target); });
}
