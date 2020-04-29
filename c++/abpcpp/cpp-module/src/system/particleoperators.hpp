
#ifndef __PARTICLEOPERATORS_HPP__
#define __PARTICLEOPERATORS_HPP__

#include "particletype.hpp"

struct reset_particle_forces
{
    inline 
    ParticleType operator()(const ParticleType& particle)
    {
        ParticleType v = particle;
        v.forceC.x = 0.0;
        v.forceC.y = 0.0;
        v.forceC.z = 0.0;
        v.forceA.x = 0.0;
        v.forceA.y = 0.0;
        v.forceA.z = 0.0;
        return v;
    }
};

struct reset_particle_torques
{
    inline
    ParticleType operator()(const ParticleType& particle)
    {
        ParticleType v = particle;
        v.tau.x = 0.0;
        v.tau.y = 0.0;
        v.tau.z = 0.0;
        return v;
    }
};

struct reset_particle_energy
{
    inline
    ParticleType operator()(const ParticleType& particle)
    {
        ParticleType v = particle;
        v.energy = 0.0;
        return v;
    }
};


struct reset_particle_forces_torques_energy
{
    inline
    ParticleType operator()(const ParticleType& particle)
    {
        ParticleType v = particle;
        v.energy = 0.0;
        v.forceC.x = 0.0;
        v.forceC.y = 0.0;
        v.forceC.z = 0.0;
        v.forceA.x = 0.0;
        v.forceA.y = 0.0;
        v.forceA.z = 0.0;
        v.tau.x = 0.0;
        v.tau.y = 0.0;
        v.tau.z = 0.0;
        return v;
    }
};
#endif