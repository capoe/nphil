#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  
#include <pybind11/stl.h>    
#include "npfga.hpp"

namespace py = pybind11;
namespace npfga = soap::npfga;

PYBIND11_MODULE(_nphil, m) {
    py::class_<npfga::FGraph>(m, "FGraph")
        .def(py::init<std::string, double, double, double>(), 
            py::arg("correlation_measure"),
            py::arg("unit_min_exp"),
            py::arg("unit_max_exp"),
            py::arg("rank_coeff"))
        .def("getNodes", &npfga::FGraph::getFNodes, py::return_value_policy::reference)
        .def("__len__", &npfga::FGraph::size)
        .def("addRootNode", &npfga::FGraph::addRootNode,
            py::arg("varname"),
            py::arg("sign"),
            py::arg("zero"),
            py::arg("scale"),
            py::arg("units"))
        .def("addLayer", &npfga::FGraph::addLayer,
            py::arg("uops"),
            py::arg("bops"))
        .def("generate", &npfga::FGraph::generate)
        .def("apply", &npfga::FGraph::applyNumpy)
        .def("applyAndCorrelate", &npfga::FGraph::applyAndCorrelateNumpy);
    py::class_<npfga::FNode>(m, "FNode")
        .def(py::init<>())
        .def_property_readonly("prefactor", &npfga::FNode::getPrefactor)
        .def_property_readonly("unit_prefactor", &npfga::FNode::getUnitPrefactor)
        .def_property_readonly("generation", &npfga::FNode::getGenerationIdx)
        .def_property_readonly("is_root", &npfga::FNode::isRoot)
        .def_property_readonly("tag", &npfga::FNode::calculateTag)
        .def_property_readonly("op_tag", &npfga::FNode::getOperatorTag)
        .def_property_readonly("expr", &npfga::FNode::getExpr)
        .def_property("cov", &npfga::FNode::getCovariance, &npfga::FNode::setCovariance)
        .def_property("q", &npfga::FNode::getConfidence, &npfga::FNode::setConfidence)
        .def("getRoots", &npfga::FNode::getRoots)
        .def("getParents", &npfga::FNode::getParents);
}
