// SPDX-License-Identifier: LGPL-3.0-or-later
#include "python_model.hpp"

#include "CollisionGeometry.hpp"
#include "GenericAgent.hpp"
#include "NeighborhoodSearch.hpp"
#include "OperationalModel.hpp"
#include "OperationalModels/CustomModel/CustomModelData.hpp"
#include "OperationalModels/CustomModel/CustomModelUpdate.hpp"
#include "SimulationError.hpp"
#include "conversion.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>

namespace py = pybind11;

namespace
{

Point requiredPosition(py::handle object)
{
    constexpr auto attribute = "position";
    if(!py::hasattr(object, attribute)) {
        throw SimulationError("Missing attribute '{}'.", attribute);
    }
    return intoPoint(py::cast<std::tuple<double, double>>(object.attr(attribute)));
}

py::object requireAttribute(py::handle object, const char* attribute)
{
    if(!py::hasattr(object, attribute)) {
        throw SimulationError(
            "CustomModelAgentUpdate returned by compute_new_position() is missing '{}'", attribute);
    }
    return object.attr(attribute);
}

std::optional<Point> optionalPointAttribute(py::handle update, const char* attribute)
{
    auto value = requireAttribute(update, attribute);
    if(value.is_none()) {
        return std::nullopt;
    }

    try {
        return intoPoint(py::cast<std::tuple<double, double>>(value));
    } catch(const py::cast_error&) {
        throw SimulationError(
            "CustomModelAgentUpdate.{} must be None or tuple[float, float]", attribute);
    }
}

} // namespace

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
    if(!py::hasattr(_model, "_compute_new_position") ||
       !py::hasattr(_model, "_check_model_constraint")) {
        throw std::invalid_argument("_PythonModel requires a CustomOperationalModel instance");
    }
}

OperationalModelUpdate PythonModel::ComputeNewPosition(
    double dT,
    const GenericAgent& agent,
    const CollisionGeometry& geometry,
    const NeighborhoodSearch<GenericAgent>& neighborhoodSearch) const
{
    py::gil_scoped_acquire gil;

    py::object pythonAgent = py::cast(agent);
    py::object pythonGeometry = py::cast(&geometry, py::return_value_policy::reference);
    py::object pythonNeighborhoodSearch = py::cast(
        const_cast<NeighborhoodSearch<GenericAgent>*>(&neighborhoodSearch),
        py::return_value_policy::reference);

    py::object update = _model.attr("compute_new_position")(
        dT, pythonAgent, pythonGeometry, pythonNeighborhoodSearch);

    return CustomModelUpdate{GilSafePyObject{std::move(update)}};
}

void PythonModel::ApplyUpdate(const OperationalModelUpdate& update, GenericAgent& agent) const
{
    py::gil_scoped_acquire gil;

    const auto& pythonUpdate = std::get<CustomModelUpdate>(update).Get<GilSafePyObject>().Get();

    agent.pos = requiredPosition(pythonUpdate);

    auto model = requireAttribute(pythonUpdate, "model");
    if(model.is_none()) {
        throw SimulationError("PythonModelData may not a python  'None'");
    }
    if(!py::hasattr(pythonUpdate, "__dict__")) {
        throw SimulationError("TODO");
    }
    auto updateFields = pythonUpdate.attr("__dict__").cast<py::dict>();
    for(auto [name, value] : updateFields) {
        if(name.cast<std::string>().compare("position") == 0) {
            continue;
        }
        if(!py::hasattr(model, name))
            throw SimulationError("data missing expected field: {}", name.cast<std::string>());
        model.attr(name) = value;
    }
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
    py::class_<OperationalModel, py::smart_holder>(m, "_OperationalModel");

    // py::class_<CustomModelData>(m, "_CustomModelData")
    //     .def(py::init([](py::object model) {
    //         return CustomModelData{GilSafePyObject{std::move(model)}};
    //     }))
    //     .def_property_readonly(
    //         "model", [](CustomModelData& data) { return data.Get<GilSafePyObject>().Get(); });

    py::class_<PythonModel, OperationalModel, py::smart_holder>(m, "_PythonModel")
        .def(py::init<py::object>(), py::arg("model"));
}
