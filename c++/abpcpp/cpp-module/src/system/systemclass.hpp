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

#include "particletype.hpp"
#include "../box/box.hpp"
#include "../box/pbc.hpp"

class SystemClass
{
    public:
    /**
     * @brief Create an empty particles list
     * @param void
    */
    SystemClass(const BoxType &box) : _box(box), Numparticles(0) {particles.clear();}
    /**
     * Create a particles from given vectors
    */
    SystemClass(const host::vector<ParticleType> &_particles, const BoxType &box): _box(box)
    {
        this->add_particle(_particles);
    }
    ~SystemClass() {}

    /**
     * @brief get the particles loaded in the system
     * @param void 
     * @return const BoxType&
     */
    const BoxType& get_box(void) {return _box;}

    /**
     * @brief get the particles loaded in the system
     * @param void 
     * @return host::vector<ParticleType> 
     */
    host::vector<ParticleType> get(void) { return (particles); }

    /**
     * @brief get the particles loaded in the system
     * @param const host::vector<ParticleType>& _particles 
     * @return void
     */
    void add_particle(const host::vector<ParticleType> &_particles)
    {
        particles = _particles;
        Numparticles = _particles.size();
    }
    /**
     * @brief add a single particle into the system
     * @param ParticleType&
     * @return void
     */
    void add_particle(const ParticleType &_particle)
    {
        particles.push_back(_particle);
        particles[Numparticles].id = Numparticles;
        Numparticles = particles.size();
    }

    void apply_periodic(void)
    {
        for (int pindex = 0; pindex < Numparticles; pindex++)
            host::enforce_periodic(particles[pindex].r, particles[pindex].ip, _box);
    }

    /** Variables **/
    host::vector<ParticleType> particles; //!< Particle list
    int Numparticles;                     //!< Number of particles

    private:
    BoxType _box;


};

#endif