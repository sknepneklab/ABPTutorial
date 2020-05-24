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
#ifndef __pybind_export_types_hpp__
#define __pybind_export_types_hpp__

#include "globaltypes.hpp"

/*  @Note In the same way that we define operators in c++ 
    (see system/particleoperators.hpp), in python is possible to define what is call
    magic methods https://rszalski.github.io/magicmethods/
    this methods are used to perfom operations in classes
    a trival example is given here for the real2 class
    (see types/pybind_export_types.hpp)
*/
void export_real2(py::module &m)
{
    py::class_<real2>(m, "real2")
        .def(py::init<>())
        .def("__init__", [](real2 &instance, double x, double y) {
            new (&instance) real2();
            instance.x = x;
            instance.y = y;
        })

        .def("__repr__",[](const real2 &a) {
            return( "<real2 x = " + to_string(a.x) + " y = " + to_string(a.y) + " >");
        })
        .def_readwrite("x", &real2::x)
        .def_readwrite("y", &real2::y)
        /*opeators*/
        .def("__mul__", [](const real2 &a, const real2& b) 
        {
            return (vdot(a,b));
        }, py::is_operator())
        .def("__abs__", [](const real2 &a) 
        {
            return (sqrt(vdot(a,a)));
        }, py::is_operator())
        .def("__add__", [](const real2 &a, const real2& b) 
        {
            real2 c;
            vsum(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__sub__", [](const real2 &a, const real2& b) 
        {
            real2 c;
            vsub(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__matmul__", [](const real2 &a, const real2& b) 
        {
            return (vcross(a,b));
        }, py::is_operator())
        .def("__neg__", [](const real2 &a) 
        {
            real2 c;
            c.x = -a.x;
            c.y = -a.y;
            return (c);
        }, py::is_operator())
        .def("__pow__", [](const real2 &a, const double & b) 
        {
            real2 c;
            c.x = pow(a.x,b);
            c.y = pow(a.y,b);
            return (c);
        }, py::is_operator())
        .def("__iadd__", [](const real2 &a, const real2& b) 
        {
            real2 c;
            vsum(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__isub__", [](const real2 &a, const real2& b) 
        {
            real2 c;
            vsub(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__imul__", [](const real2 &a, const real2& b) 
        {
            return (vdot(a,b));
        }, py::is_operator())
        .def("__scale__", [](const real2 &a, const real& b) 
        {   
            real2 c = a;
            c.x*=b;
            c.y*=b;
            return c;
        }, py::is_operator())
        ;
}

void export_inth2(py::module &m)
{
    py::class_<inth2>(m, "int2")
        .def(py::init<>())
        .def("__init__", [](inth2 &instance, int x, int y) {
            new (&instance) inth2();
            instance.x = x;
            instance.y = y;
        })

        .def("__repr__",[](const inth2 &a) {
            return( "<int2 x = " + to_string(a.x) + " y = " + to_string(a.y) + " >");
        })
        .def_readwrite("x", &inth2::x)
        .def_readwrite("y", &inth2::y)
        ;
}

void export_bool2(py::module &m)
{
    py::class_<bool2>(m, "bool2")
        .def(py::init<>())
        .def("__init__", [](bool2 &instance, bool x, bool y) {
            new (&instance) bool2();
            instance.x = x;
            instance.y = y;
        })

        .def("__repr__",[](const bool2 &a) {
            return( "<bool2 x = " + to_string(a.x) + " y = " + to_string(a.y) + " >");
        })
        .def_readwrite("x", &bool2::x)
        .def_readwrite("y", &bool2::y)
        ;
}
#endif
