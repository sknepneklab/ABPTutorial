#include "polar_align.hpp"

void PolarAlign::compute_energy(void)
{
    for (int pindex_i = 0; pindex_i < _system.Numparticles; pindex_i ++)
    {
        ParticleType pi = _system.particles[pindex_i];
        //loop over the neighbours particles
        for(int c = 0; c < pi.coordination; c ++)
        {
            int pindex_j = c + _neighbourslist.max_ng_per_particle * pindex_i;
            ParticleType pj = _system.particles[pindex_j];

            real2 rij = host::minimum_image(pi.r, pj.r, _system.get_box());
            real lendr = vdot(rij, rij);
            if (lendr <= a*a)
            {
                pi.energy+=J*vdot(pi.n, pj.n);
            }
        }
        //put back the particle in the list
        _system.particles[pindex_i] = pi;
    }

}

void PolarAlign::compute(void)
{
    for (int pindex_i = 0; pindex_i < _system.Numparticles; pindex_i ++)
    {
        ParticleType pi = _system.particles[pindex_i];
        //loop over the neighbours particles
        for(int c = 0; c < pi.coordination; c ++)
        {
            int pindex_j = _neighbourslist.nglist[c + _neighbourslist.max_ng_per_particle * pindex_i];
            const ParticleType pj = _system.particles[pindex_j];
            real2 rij = host::minimum_image(pi.r, pj.r, _system.get_box());
            real lendr = vdot(rij, rij);
            if (lendr <= a*a)
            {
                pi.tau+=J*vcross(pi.n, pj.n);
            }
        }
        //put back the particle in the list
        _system.particles[pindex_i] = pi;
    }
    /**/
}
