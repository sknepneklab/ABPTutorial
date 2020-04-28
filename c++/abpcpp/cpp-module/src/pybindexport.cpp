#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>

namespace py = pybind11;

//types
#include "types/pybind_export_types.hpp"

//particles
#include "particle/pybind_export_particles.hpp"

//box
#include "box/pybind_export_box.hpp"

PYBIND11_MODULE(abpcpp, m)
{
    ///Documentation
    m.doc() = R"pbdoc(
        KITP20 Tutorials
        -----------------------
        .. currentmodule:: abpcpp
        .. autosummary::
           :toctree: _generate
           compute_volume
           subtract
    )pbdoc";
    m.attr("__version__") = "1.0a";
    ///redirect std::cout and std::cerr
    add_ostream_redirect(m, "ostream_redirect");
    export_real3(m);
    PYBIND11_NUMPY_DTYPE(real3, x, y, z);
    //export_int3(m);
    //PYBIND11_NUMPY_DTYPE(int3, x, y, z);
    export_ParticleType(m);
    export_ParticleType_Vector(m);
    export_BoxType(m);
}