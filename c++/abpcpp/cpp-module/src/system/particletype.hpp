
#ifndef __PARTICLETYPE_HPP__
#define __PARTICLETYPE_HPP__

#include "../types/globaltypes.hpp"

struct ParticleType
{
    int id;           //!< particle id
    int type;         //!< Type
    real radius;      //!< radius of the particle
    real3 r;          //!< Position in the embedding 3d flat space
    inth3 ip;         //!< Periodic box image flags (to enable unwrapping of particle coordinates)
    real3 n;          //!< Particle direction vector (not necessarily equal to velocity direction)
    int coordination; //!< Keeps track of the number of neighbours
    inth3 cellId;     //!< CellId that belongs for linked list
    real3 v;          //!< Velocity
    real3 forceC;     //!< conservative force
    real3 forceA;     //!< active force
    real energy;      //!< conservative energy
};

#endif