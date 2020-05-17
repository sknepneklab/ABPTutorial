#include "selfpropulsion.hpp"

void SelfPropulsionForce::compute(void)
{
    for (int pindex_i = 0; pindex_i < _system.Numparticles; pindex_i ++)
    {
        ParticleType pi = _system.particles[pindex_i];
        pi.forceC.x+=alpha*pi.n.x;
        pi.forceC.y+=alpha*pi.n.y;
        //put back the particle in the list
        _system.particles[pindex_i] = pi;
    }
}
