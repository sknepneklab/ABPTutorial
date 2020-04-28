#ifndef __PARTICLES_HPP__
#define __PARTICLES_HPP__

#include "particle.hpp"

struct ParticlesClass
{
    /**
     * @brief Create an empty particles list
     * @param void
    */
    ParticlesClass() : Numparticles(0) {particles.clear();}
    /**
     * Create a particles from given vectors
    */
    ParticlesClass(const host::vector<ParticleType> &_particles //!< Particle list
                   )
    {
        this->set(_particles);
    }
    ~ParticlesClass() {}
    /**
     * @brief get the vertices loaded in the system
     * @param void 
     * @return host::vector<ParticleType> 
     */
    host::vector<ParticleType> get(void) { return (particles); }

    /**
     * @brief get the vertices loaded in the system
     * @param const host::vector<ParticleType>& _particles 
     * @return void
     */
    void set(const host::vector<ParticleType> &_particles)
    {
        particles = _particles;
        Numparticles = _particles.size();
    }
    /**
     * @brief add a single particle into the system
     * @param ParticleType&
     * @return void
     */
    void add_particle(ParticleType &_particle)
    {
        _particle.id = Numparticles;
        particles.push_back(_particle);
        Numparticles = particles.size();
    }

    host::vector<ParticleType> particles; //!< Particle list
    int Numparticles;                     //!< Number of particles
};

#endif