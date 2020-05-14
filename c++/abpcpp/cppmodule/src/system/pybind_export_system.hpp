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
#ifndef __pybind_export_system_hpp__
#define __pybind_export_system_hpp__

#include "particletype.hpp"
#include "systemclass.hpp"

void export_ParticleType(py::module &m)
{
    py::class_<ParticleType>(m, "Particle")
        .def(py::init<>())
        .def_readwrite("id", &ParticleType::id)
        .def_readwrite("r", &ParticleType::r)
        .def_readwrite("ip", &ParticleType::ip)
        .def_readwrite("n", &ParticleType::n)
        .def_readwrite("coordination", &ParticleType::coordination)
        .def_readonly("cellId", &ParticleType::cellId)
        .def_readwrite("type",  &ParticleType::type, "Material type")
        .def_readonly("v", &ParticleType::v, "Particle velocity")
        .def_readonly("forceC", &ParticleType::forceC, "Particle Conservative Force")
        .def_readonly("energy", &ParticleType::energy, "Particle Conservative Energy")
        .def_readwrite("radius", &ParticleType::radius, "Particle radius");
}

void export_ParticleType_Vector(py::module &m)
{
    py::class_<std::vector<ParticleType>>(m, "ParticleVector")
        .def(py::init<>())
        .def("clear", &std::vector<ParticleType>::clear)
        .def("pop_back", &std::vector<ParticleType>::pop_back)
        .def("append", (void (std::vector<ParticleType>::*)(const ParticleType &)) & std::vector<ParticleType>::push_back)
        .def("__len__", [](const std::vector<ParticleType> &v) { return v.size(); })
        .def("__iter__", [](std::vector<ParticleType> &v) { return py::make_iterator(v.begin(), v.end()); }, py::keep_alive<0, 1>())
        .def("__getitem__", [](const std::vector<ParticleType> &v, size_t i) { if (i >= v.size()) throw py::index_error(); return v[i]; })
        .def("__setitem__", [](std::vector<ParticleType> &v, size_t i, ParticleType &d) { if (i >= v.size()) throw py::index_error(); v[i] = d; });
}

void export_SytemClass(py::module &m)
{
    py::class_<SystemClass>(m, "System")
        .def(py::init<const BoxType &>())
        .def(py::init<const host::vector<ParticleType> &, const BoxType &>())
        .def("get_particles", &SystemClass::get)
        .def("add", (void (SystemClass::*)(const ParticleType &)) &SystemClass::add_particle)
        .def("add", (void (SystemClass::*)(const host::vector<ParticleType> &)) &SystemClass::add_particle)
        .def("apply_periodic", &SystemClass::apply_periodic)
        .def("box", &SystemClass::get_box)
        ;
}

#endif