// SPDX-License-Identifier: LGPL-3.0-or-later
#include "python_model.hpp"

#include "AgentRouting.hpp"
#include "AgentState.hpp"
#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/CustomModel/CustomModelState.hpp"
#include "OperationalModels/OperationalModelType.hpp"
#include "SimulationError.hpp"
#include "conversion.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <tuple>
#include <utility>

namespace py = pybind11;

GilSafePyObject::GilSafePyObject(py::object obj) : _obj(std::move(obj))
{
}

GilSafePyObject::GilSafePyObject(const GilSafePyObject& other)
{
    py::gil_scoped_acquire gil;
    _obj = other._obj; // share by reference (incref), do not clone
}

GilSafePyObject& GilSafePyObject::operator=(const GilSafePyObject& other)
{
    if(this == &other) {
        return *this;
    }

    py::gil_scoped_acquire gil;
    _obj = other._obj; // share by reference (incref), do not clone
    return *this;
}

GilSafePyObject& GilSafePyObject::operator=(GilSafePyObject&& other) noexcept
{
    if(this == &other) {
        return *this;
    }

    // Move-assignment decrefs the previously held object, so it must hold the GIL.
    py::gil_scoped_acquire gil;
    _obj = std::move(other._obj);
    return *this;
}

GilSafePyObject::~GilSafePyObject()
{
    py::gil_scoped_acquire gil;
    _obj = py::object();
}

const py::object& GilSafePyObject::Get() const
{
    return _obj;
}

py::object& GilSafePyObject::Get()
{
    return _obj;
}

void GilSafePyObject::Set(py::object obj)
{
    py::gil_scoped_acquire gil;
    _obj = std::move(obj);
}

PythonModel::PythonModel(py::object model) : _model(std::move(model))
{
    py::gil_scoped_acquire gil;
    if(!_model || _model.is_none()) {
        throw std::invalid_argument("_PythonModel requires a CustomOperationalModel instance");
    }
    if(!py::hasattr(_model, "_compute_next") ||
       !py::hasattr(_model, "_check_model_constraint")) {
        throw std::invalid_argument("_PythonModel requires a CustomOperationalModel instance");
    }
}

void PythonModel::ComputeNext(
    double dT,
    const AgentState& current,
    AgentState& next,
    const AgentRouting& routing,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    py::gil_scoped_acquire gil;

    py::object currentPyState =
        std::get<CustomModelState>(*current.extras).Get<GilSafePyObject>().Get();

    py::object pythonGeometry = py::cast(&geometry, py::return_value_policy::reference);
    py::object pythonNeighborhoodSearch = py::cast(
        const_cast<NeighborhoodSearch<GenericAgent>*>(&neighborhoodSearch),
        py::return_value_policy::reference);

    py::object pythonUpdate = _model.attr("_compute_next")(
        dT, currentPyState, routing, pythonGeometry, pythonNeighborhoodSearch);

    // "next" shares the Python state object with "current" (GilSafePyObject copies are
    // refcounted, not cloned), so this also rejects returning the current state instance.
    auto& customState = std::get<CustomModelState>(*next.extras).Get<GilSafePyObject>();
    if(pythonUpdate.is(customState.Get())) {
        throw SimulationError(
            "Current and updated model state are the same instance. "
            "compute_next() must return a new state object, "
            "e.g. dataclasses.replace(ped.model, ...).");
    }

    constexpr auto attr_name = "position";
    py::object attr;
    try {
        attr = pythonUpdate.attr(attr_name);
    } catch(const py::error_already_set& ex) {
        if(ex.matches(PyExc_AttributeError)) {
            throw SimulationError(
                "State returned by compute_next() is missing the '{}' attribute.", attr_name);
        }
        throw;
    }

    try {
        next.position = intoPoint(py::cast<std::tuple<double, double>>(attr));
    } catch(const py::cast_error&) {
        std::string actualType = "<unknown>";
        std::string valueRepr = "<unprintable>";
        try {
            actualType = std::string(py::str(py::type::of(attr).attr("__name__")));
        } catch(const py::error_already_set&) {
        }
        try {
            valueRepr = std::string(py::repr(attr));
        } catch(const py::error_already_set&) {
        }

        throw SimulationError(
            "State returned by compute_next() has attribute '{}' of wrong type: "
            "expected tuple[float, float], got {} ({})",
            attr_name,
            actualType,
            valueRepr);
    }

    // Mirror optional cross-model fields so C++ built-in models can interact
    // with Python custom model agents as proper neighbors.
    if(py::hasattr(pythonUpdate, "radius")) {
        try {
            next.radius = py::cast<double>(pythonUpdate.attr("radius"));
        } catch(const py::cast_error&) {
        }
    }
    if(py::hasattr(pythonUpdate, "velocity")) {
        try {
            next.velocity =
                intoPoint(py::cast<std::tuple<double, double>>(pythonUpdate.attr("velocity")));
        } catch(const py::cast_error&) {
        }
    }
    customState.Set(pythonUpdate);
}

void PythonModel::CheckModelConstraint(
    const GenericAgent& agent,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch,
    const CollisionGeometry& geometry) const
{
    py::gil_scoped_acquire gil;

    py::object pythonAgent = py::cast(agent);
    py::object pythonNeighborhoodSearch = py::cast(
        const_cast<NeighborhoodSearch<GenericAgent>*>(&neighborhoodSearch),
        py::return_value_policy::reference);
    py::object pythonGeometry = py::cast(&geometry, py::return_value_policy::reference);

    _model.attr("_check_model_constraint")(pythonAgent, pythonNeighborhoodSearch, pythonGeometry);
}

void init_python_model(py::module_& m)
{
    // Helper for calling a built-in C++ model from within a Python custom-model callback.
    //
    // Takes a prototype GenericAgent (to borrow id, journey, stage, and navigation target),
    // copies it, runs ComputeNext on the copy, and returns the resulting AgentState.
    // The caller can then extract the new position (and any other updated fields) and
    // return a new Python state from their compute_next() implementation.
    //
    // Before calling, mirror your Python state's fields (desired_speed, radius, etc.) onto
    // agent._native.model so the C++ model sees the correct per-agent parameters.
    //
    // Usage from Python:
    //   # set parameters on the native model state first
    //   agent._native.model.desired_speed = state.desired_speed
    //   next_result = py_jps._builtin_compute_next(
    //       model, dt, agent._native, geometry._obj, neighborhood_search._obj)
    //   new_position = next_result.position
    m.def(
        "_builtin_compute_next",
        [](const OperationalModel& model,
           double dt,
           GenericAgent prototype,
           const CollisionGeometry& geometry,
           const NeighborhoodSearch<GenericAgent>& ns) -> AgentState {
            prototype.routing.destination = prototype.routing.target;
            GenericAgent next = prototype;
            model.ComputeNext(dt, prototype.model, next.model, prototype.routing, geometry, ns);
            return next.model;
        },
        py::arg("model"),
        py::arg("dt"),
        py::arg("prototype"),
        py::arg("geometry"),
        py::arg("neighborhood_search"));

    py::class_<OperationalModel, py::smart_holder>(m, "OperationalModel");

    // Factory that wraps a Python custom model state in an AgentState for use with
    // Simulation.add_agent(). The position and optional radius/velocity are mirrored
    // into the AgentState fields so C++ built-in neighbor models can read them.
    // Factory that wraps a Python custom model state in an AgentState for use with
    // Simulation.add_agent(). The Python state is stored as CustomState inside extras.
    // Position and optional radius/velocity are mirrored into common AgentState fields
    // so C++ built-in neighbor models can read them without touching the Python payload.
    m.def(
        "_CustomModelData",
        [](py::object model) -> AgentState {
            const auto position =
                intoPoint(py::cast<std::tuple<double, double>>(model.attr("position")));
            AgentState state;
            state.type = OperationalModelType::CUSTOM_MODEL;
            state.position = position;
            if(py::hasattr(model, "radius")) {
                try {
                    state.radius = py::cast<double>(model.attr("radius"));
                } catch(const py::cast_error&) {
                }
            }
            if(py::hasattr(model, "velocity")) {
                try {
                    state.velocity =
                        intoPoint(py::cast<std::tuple<double, double>>(model.attr("velocity")));
                } catch(const py::cast_error&) {
                }
            }
            state.extras = CustomModelState{GilSafePyObject{std::move(model)}};
            return state;
        },
        py::arg("model"));

    py::class_<PythonModel, OperationalModel, py::smart_holder>(m, "_PythonModel")
        .def(py::init<py::object>(), py::arg("model"));
}
