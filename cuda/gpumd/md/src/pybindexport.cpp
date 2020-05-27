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
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>

namespace py = pybind11;

//types
#include "types/pybind_export_types.hpp"

//box
#include "box/pybind_export_box.hpp"

//system
#include "system/pybind_export_system.hpp"

//compute
#include "evolver/pybind_export_evolver.hpp"
//
#include "configuration/pybind_export_gpu_configuration.hpp"

// Json Reader
#include "pybind_export_from_json.hpp"

PYBIND11_MODULE(nvccmodule, m)
{
    ///Documentation
    m.doc() = R"pbdoc(
        KITP20 Tutorials
        -----------------------
        .. currentmodule:: nvccmodule
        .. autosummary::
           :toctree: _generate
    )pbdoc";
    m.attr("__version__") = "1.0a";
    ///redirect std::cout and std::cerr
    add_ostream_redirect(m, "ostream_redirect");
    export_real2(m);
    export_inth2(m);
    export_bool2(m);
    PYBIND11_NUMPY_DTYPE(real2, x, y);
    PYBIND11_NUMPY_DTYPE(inth2, x, y);
    PYBIND11_NUMPY_DTYPE(bool2, x, y);
    //gpu
    export_ExecutionPolicyGPU(m);
    //md
    export_BoxType(m);
    export_ParticleType(m);
    export_ParticleType_Vector(m);
    export_SystemClass(m);
    export_EvolverClass(m);
    //json reader
    export_FastJsonReader(m);
}
