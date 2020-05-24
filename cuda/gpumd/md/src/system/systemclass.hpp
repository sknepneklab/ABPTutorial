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
#ifndef __systemclass_hpp__
#define __systemclass_hpp__

#include "../configuration/execution_policy.hpp"
#include "../types/hostvector.hpp"
#include "../types/devicevector.hpp"
#include "../box/box.hpp"
#include "../box/pbc.hpp"
#include "particletype.hpp"

class SystemClass
{
public:
    SystemClass(const BoxType &box) : _box(box), Numparticles(0) {}
    /**
     * Create a particles from given vectors
    */
    SystemClass(host::vector<ParticleType> &_particles, const BoxType &box) : _box(box)
    {
        this->add_particle(_particles);
    } //!< constructor
    ~SystemClass() {}
    /**
     * @brief get the particles loaded in the system
     * @param void 
     * @return const BoxType&
     */
    const BoxType &get_box(void) { return _box; }
    /**
     * @brief set GPU execution policies
     * @param ExecutionPolicy& ep
     */
    void set_execution_policies(ExecutionPolicy &ep)
    {
        _ep = ep;
    }
    /**
     * @brief get GPU execution policies
     * @return ExecutionPolicy ep
     */
    ExecutionPolicy &get_execution_policies(void)
    {
        return (_ep);
    }
    /**
     * @brief get the vertices loaded in the system
     * @param void 
     * @return host::vector<ParticleType> 
     */
    host::vector<ParticleType> get(void) { return (device::copy(particles)); }
    /**
     * @brief get the vertices loaded in the system
     * @param host::vector<ParticleType>& _particles 
     * @return  
     */
    void add_particle(host::vector<ParticleType> &_particles)
    {
        particles = _particles;
        Numparticles = _particles.size();
    }

    void apply_periodic(void)
    {
        //for (int pindex = 0; pindex < Numparticles; pindex++)
        //    host::enforce_periodic(particles[pindex].r, particles[pindex].ip, _box);
    }
    //private:
    device::vector<ParticleType> particles; //!< Vertex list
    int Numparticles;                       //!< Number of vertices
    ///CUDA
    ExecutionPolicy _ep;

private:
    BoxType _box;
};

#endif