
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
#ifndef __particletype_hpp__
#define __particletype_hpp__

#include "../types/globaltypes.hpp"

struct ParticleType
{
    int id;           //!< particle id
    int type;         //!< Type
    real radius;      //!< radius of the particle
    real2 r;          //!< Position in the embedding 3d flat space
    inth2 ip;         //!< Periodic box image flags (to enable unwrapping of particle coordinates)
    real2 n;          //!< Particle direction vector (not necessarily equal to velocity direction)
    int coordination; //!< Keeps track of the number of neighbours
    inth2 cellId;     //!< CellId that belongs for linked list
    real2 v;          //!< Velocity
    real2 forceC;     //!< conservative force
    real2 forceA;     //!< active force
    real tau;         //!< torque
    real energy;      //!< conservative energy
};


#endif