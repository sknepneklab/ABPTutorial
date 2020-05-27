
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
#ifndef __PARTICLEOPERATORS_hpp__
#define __PARTICLEOPERATORS_hpp__

#include "particletype.hpp"
namespace host
{
    struct reset_particle_forces
    {
        inline ParticleType operator()(const ParticleType &particle)
        {
            ParticleType v = particle;
            v.forceC.x = 0.0;
            v.forceC.y = 0.0;
            v.forceA.x = 0.0;
            v.forceA.y = 0.0;
            return v;
        }
    };

    struct reset_particle_torques
    {
        inline ParticleType operator()(const ParticleType &particle)
        {
            ParticleType v = particle;
            v.tau = 0.0;
            return v;
        }
    };

    struct reset_particle_energy
    {
        inline ParticleType operator()(const ParticleType &particle)
        {
            ParticleType v = particle;
            v.energy = 0.0;
            return v;
        }
    };

    struct reset_particle_forces_torques_energy
    {
        inline ParticleType operator()(const ParticleType &particle)
        {
            ParticleType v = particle;
            v.energy = 0.0;
            v.forceC.x = 0.0;
            v.forceC.y = 0.0;
            v.forceA.x = 0.0;
            v.forceA.y = 0.0;
            v.tau = 0.0;
            return v;
        }
    };
} // namespace host
#endif