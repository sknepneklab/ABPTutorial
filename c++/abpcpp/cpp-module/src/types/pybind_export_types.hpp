#ifndef __pybind_export_types_hpp__
#define __pybind_export_types_hpp__

#include "globaltypes.hpp"

/*  @Note In the same way that we define operators in c++ 
    (see system/particleoperators.hpp), in python is possible to define what is call
    magic methods https://rszalski.github.io/magicmethods/
    this methods are used to perfom operations in classes
    a trival example is given here for the real3 class
    (see types/pybind_export_types.hpp)
*/
void export_real3(py::module &m)
{
    py::class_<real3>(m, "real3")
        .def(py::init<>())
        .def("__init__", [](real3 &instance, double x, double y, double z) {
            new (&instance) real3();
            instance.x = x;
            instance.y = y;
            instance.z = z;
        })

        .def("__repr__",[](const real3 &a) {
            return( "<real3 x = " + to_string(a.x) + " y = " + to_string(a.y) + " z = " + to_string(a.z) + " >");
        })
        .def_readwrite("x", &real3::x)
        .def_readwrite("y", &real3::y)
        .def_readwrite("z", &real3::z)
        /*opeators*/
        .def("__mul__", [](const real3 &a, const real3& b) 
        {
            return (vdot(a,b));
        }, py::is_operator())
        .def("__abs__", [](const real3 &a) 
        {
            return (sqrt(vdot(a,a)));
        }, py::is_operator())
        .def("__add__", [](const real3 &a, const real3& b) 
        {
            real3 c;
            vsum(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__sub__", [](const real3 &a, const real3& b) 
        {
            real3 c;
            vsub(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__matmul__", [](const real3 &a, const real3& b) 
        {
            real3 c;
            vcross(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__neg__", [](const real3 &a) 
        {
            real3 c;
            c.x = -a.x;
            c.y = -a.y;
            c.z = -a.z;
            return (c);
        }, py::is_operator())
        .def("__pow__", [](const real3 &a, const double & b) 
        {
            real3 c;
            c.x = pow(a.x,b);
            c.y = pow(a.y,b);
            c.z = pow(a.z,b);
            return (c);
        }, py::is_operator())
        .def("__iadd__", [](const real3 &a, const real3& b) 
        {
            real3 c;
            vsum(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__isub__", [](const real3 &a, const real3& b) 
        {
            real3 c;
            vsub(c,a,b);
            return (c);
        }, py::is_operator())
        .def("__imul__", [](const real3 &a, const real3& b) 
        {
            return (vdot(a,b));
        }, py::is_operator())
        .def("__scale__", [](const real3 &a, const real& b) 
        {   
            real3 c = a;
            c.x*=b;
            c.y*=b;
            c.z*=b;
            return c;
        }, py::is_operator())
        ;
}

#endif