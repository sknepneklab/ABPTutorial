/************************************************************************************
* MIT License                                                                       *
*                                                                                   *
* Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           *
*               fdamatoz@gmail.com                                                  *
* Permission is hereby granted, free of charge, to any person obtaining a copy      *
* of this software and associated documentation files (the "Software"), to deal     *
* in the Software without restriction, including without limitation the rights      *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         *
* copies of the Software, and to permit persons to whom the Software is             *
* furnished to do so, subject to the following conditions:                          *
*                                                                                   *
* The above copyright notice and this permission notice shall be included in all    *
* copies or substantial portions of the Software.                                   *
*                                                                                   *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     *
* SOFTWARE.                                                                         *
*************************************************************************************/

#ifndef __pybind_export_evolver_hpp__
#define __pybind_export_evolver_hpp__

#include "evolverclass.hpp"

void export_EvolverClass(py::module &m)
{
    py::class_<EvolverClass>(m, "Evolver")
        .def(py::init<SystemClass &>())
        .def("add_force", &EvolverClass::add_force)
        .def("add_torque", &EvolverClass::add_torque)
        .def("reset_forces", &EvolverClass::reset_forces)
        .def("compute_forces", &EvolverClass::compute_forces)
        .def("reset_torques", &EvolverClass::reset_torques)
        .def("compute_torque", &EvolverClass::compute_torques)
        .def("reset_energy", &EvolverClass::reset_energy)
        .def("compute_energy", &EvolverClass::compute_energy)
        .def("add_integrator", &EvolverClass::add_integrator)
        .def("set_time_step", &EvolverClass::set_time_step)
        .def("set_global_temperature", &EvolverClass::set_global_temperature)
        .def("evolve", &EvolverClass::evolve)
        .def("create_neighbourlist",&EvolverClass::create_neighbourlist)
        .def("fill_neighbourlist",&EvolverClass::fill_neighbourlist)
        .def("get_neighbourlist",&EvolverClass::get_neighbourlist)
        ;
}
#endif
