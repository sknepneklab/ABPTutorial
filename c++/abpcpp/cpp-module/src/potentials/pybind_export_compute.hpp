#ifndef __pybind_export_compute_hpp__
#define __pybind_export_compute_hpp__

#include "computeclass.hpp"

void export_ComputeClass(py::module &m)
{
    py::class_<ComputeClass>(m, "Compute")
        .def(py::init<SystemClass &>())
        .def("add_force", &ComputeClass::add_force)
        .def("add_force", &ComputeClass::add_torque)
        .def("reset_forces", &ComputeClass::reset_forces)
        .def("compute_forces", &ComputeClass::compute_forces)
        .def("reset_torques", &ComputeClass::reset_torques)
        .def("compute_torque", &ComputeClass::compute_torque)
        .def("reset_energy", &ComputeClass::reset_energy)
        .def("compute_energy", &ComputeClass::compute_energy)
        ;
}
#endif
