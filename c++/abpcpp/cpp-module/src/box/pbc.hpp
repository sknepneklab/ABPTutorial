#ifndef __PBC_HPP__
#define __PBC_HPP__

#include "../types/globaltypes.hpp"
#include "box.hpp"

namespace host
{
inline real3 minimum_image(const real3 &ri,
                           const real3 &rj,
                           const BoxType &box)
{
    real3 rij;
    vsub(rij, rj, ri);
    if (box.periodic.x)
    {
        if (rij.x > box.Lhi.x)
            rij.x -= box.L.x;
        else if (rij.x < box.Llo.x)
            rij.x += box.L.x;
    }
    if (box.periodic.y)
    {
        if (rij.y > box.Lhi.y)
            rij.y -= box.L.y;
        else if (rij.y < box.Llo.y)
            rij.y += box.L.y;
    }
    if (box.periodic.z)
    {
        if (rij.z > box.Lhi.z)
            rij.z -= box.L.z;
        else if (rij.z < box.Llo.z)
            rij.z += box.L.z;
    }
    return rij;
}

inline real3 enforce_periodic(const real3 &r,
                              const BoxType &box)
{
    real3 _r = r;
    if (box.periodic.x)
    {
        if (_r.x > box.Lhi.x)
            _r.x -= box.L.x;
        else if (_r.x < box.Llo.x)
            _r.x += box.L.x;
    }
    if (box.periodic.y)
    {
        if (_r.y > box.Lhi.y)
            _r.y -= box.L.y;
        else if (_r.y < box.Llo.y)
            _r.y += box.L.y;
    }
    if (box.periodic.z)
    {
        if (_r.z > box.Lhi.z)
            _r.z -= box.L.z;
        else if (_r.z < box.Llo.z)
            _r.z += box.L.z;
    }
    return _r;
}

} // namespace host

#endif
