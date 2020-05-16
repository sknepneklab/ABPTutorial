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
#ifndef __pbc_hpp__
#define __pbc_hpp__

#include "../types/globaltypes.hpp"
#include "box.hpp"

namespace host
{
inline real2 minimum_image(const real2 &ri,
                           const real2 &rj,
                           const BoxType &box)
{
    real2 rij;
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
    return rij;
}

inline real2 enforce_periodic(const real2 &r,
                              const BoxType &box)
{
    real2 _r = r;
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
    return _r;
}

inline void enforce_periodic(real2 &r,
                             inth2 &ip,
                             const BoxType &box)
{
    if (box.periodic.x)
    {
        if (r.x <= box.Llo.x)
        {
            r.x += box.L.x;
            ip.x--;
        }
        else if (r.x > box.Lhi.x)
        {
            r.x -= box.L.x;
            ip.x++;
        }
    }
    if (box.periodic.y)
    {
        if (r.y <= box.Llo.y)
        {
            r.y += box.L.y;
            ip.y--;
        }
        else if (r.y > box.Lhi.y)
        {
            r.y -= box.L.y;
            ip.y++;
        }
    }
}

} // namespace host

#endif
