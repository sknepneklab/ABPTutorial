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
#ifndef __pybind_export_box_hpp__
#define __pybind_export_box_hpp__

#include "box.hpp"

void export_BoxType(py::module &m)
{
    py::class_<BoxType>(m, "box")
        .def(py::init<>())
        .def("__init__", [](BoxType &self,
                            double Lx,
                            double Ly,
                            double Lz) {
            new (&self) BoxType();
            self.L.x = fabs(Lx);
            self.L.y = fabs(Ly);
            self.L.z = fabs(Lz);
            self.Llo.x = -0.5 * self.L.x;
            self.Lhi.x = 0.5 * self.L.x;
            self.Llo.y = -0.5 * self.L.y;
            self.Lhi.y = 0.5 * self.L.y;
            self.Llo.z = -0.5 * self.L.z;
            self.Lhi.z = 0.5 * self.L.z;
            self.periodic.x = true;
            self.periodic.y = true;
            self.periodic.z = true;
        })
        .def("__init__", [](BoxType &self, std::pair<double, double> Lxpair, std::pair<double, double> Lypair, std::pair<double, double> Lzpair) {
            new (&self) BoxType();
            double Lxlo = std::get<0>(Lxpair);
            double Lxhi = std::get<1>(Lxpair);
            double Lylo = std::get<0>(Lypair);
            double Lyhi = std::get<1>(Lypair);
            double Lzlo = std::get<0>(Lzpair);
            double Lzhi = std::get<1>(Lzpair);
            assert(Lxlo < Lxhi);
            assert(Lylo < Lyhi);
            assert(Lzlo < Lzhi);
            self.L.x = Lxhi - Lxlo;
            self.L.y = Lyhi - Lylo;
            self.L.z = Lzhi - Lzlo;
            self.Llo.x = Lxlo;
            self.Lhi.x = Lxhi;
            self.Llo.y = Lylo;
            self.Lhi.y = Lyhi;
            self.Llo.z = Lzlo;
            self.Lhi.z = Lzhi;
            self.periodic.x = true;
            self.periodic.y = true;
            self.periodic.z = true;
        })
        .def("__init__", [](BoxType &self, double Lx, double Ly, double Lz, bool periodic_x, bool periodic_y, bool periodic_z) {
            new (&self) BoxType();
            self.L.x = fabs(Lx);
            self.L.y = fabs(Ly);
            self.L.z = fabs(Lz);
            self.Llo.x = -0.5 * self.L.x;
            self.Lhi.x = 0.5 * self.L.x;
            self.Llo.y = -0.5 * self.L.y;
            self.Lhi.y = 0.5 * self.L.y;
            self.Llo.z = -0.5 * self.L.z;
            self.Lhi.z = 0.5 * self.L.z;
            self.periodic.x = periodic_x;
            self.periodic.y = periodic_y;
            self.periodic.z = periodic_z;
        })
        .def("__init__", [](BoxType &self, std::pair<bool, std::pair<double, double>> Lxpair, std::pair<bool, std::pair<double, double>> Lypair, std::pair<bool, std::pair<double, double>> Lzpair) {
            new (&self) BoxType();
            double Lxlo = std::get<0>(std::get<1>(Lxpair));
            double Lxhi = std::get<1>(std::get<1>(Lxpair));
            double Lylo = std::get<0>(std::get<1>(Lypair));
            double Lyhi = std::get<1>(std::get<1>(Lypair));
            double Lzlo = std::get<0>(std::get<1>(Lzpair));
            double Lzhi = std::get<1>(std::get<1>(Lzpair));
            assert(Lxlo < Lxhi);
            assert(Lylo < Lyhi);
            assert(Lzlo < Lzhi);
            self.L.x = Lxhi - Lxlo;
            self.L.y = Lyhi - Lylo;
            self.L.z = Lzhi - Lzlo;
            self.Llo.x = Lxlo;
            self.Lhi.x = Lxhi;
            self.Llo.y = Lylo;
            self.Lhi.y = Lyhi;
            self.Llo.z = Lzlo;
            self.Lhi.z = Lzhi;
            self.periodic.x = std::get<0>(Lxpair);
            self.periodic.y = std::get<0>(Lypair);
            self.periodic.z = std::get<0>(Lzpair);
        })
        .def("__repr__", [](const BoxType &self) {
            auto return_val = "<box Lx = " + to_string(self.L.x) + " Ly = " + to_string(self.L.y) + " Lz = " + to_string(self.L.z) + " \n";
            return_val+= "<box Lx = (" + to_string(self.Llo.x) + ", " + to_string(self.Lhi.x) + ")\n";
            return_val+= "<box Ly = (" + to_string(self.Llo.y) + ", " + to_string(self.Lhi.y) + ")\n";
            return_val+= "<box Lz = (" + to_string(self.Llo.z) + ", " + to_string(self.Lhi.z) + ")\n";
            return_val += "periodic_Lx = " + to_string(self.periodic.x) + " periodic_Ly = " + to_string(self.periodic.y) + " periodic_Lz = " + to_string(self.periodic.z) + " >";
            return (return_val);
        })
        .def_readwrite("Lhi", &BoxType::Lhi)
        .def_readwrite("Llo", &BoxType::Llo)
        .def_readwrite("L", &BoxType::L)
        .def_readwrite("periodic", &BoxType::periodic);
}

#endif