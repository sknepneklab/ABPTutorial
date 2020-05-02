#ifndef __pybind_export_evoler_hpp__
#define __pybind_export_evoler_hpp__

#include "evolverclass.hpp"

void export_EvolverClass(py::module &m)
{
    py::class_<EvolverClass>(m, "Compute")
        .def(py::init<SystemClass &>())
        .def("add_force", &EvolverClass::add_force)
        .def("add_force", &EvolverClass::add_torque)
        .def("reset_forces", &EvolverClass::reset_forces)
        .def("compute_forces", &EvolverClass::compute_forces)
        .def("reset_torques", &EvolverClass::reset_torques)
        .def("compute_torque", &EvolverClass::compute_torque)
        .def("reset_energy", &EvolverClass::reset_energy)
        .def("compute_energy", &EvolverClass::compute_energy)
        ;
}
#endif
