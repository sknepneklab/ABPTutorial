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

            real3 rij = host::minimum_image(pi.r, pj.r, _system.get_box());
            real lendr = sqrt(vdot(rij, rij));
            if (lendr <= a)
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
            int pindex_j = c + _neighbourslist.max_ng_per_particle * pindex_i;
            ParticleType pj = _system.particles[pindex_j];

            real3 rij = host::minimum_image(pi.r, pj.r, _system.get_box());
            real lendr = sqrt(vdot(rij, rij));
            if (lendr <= a)
            {
                real3 tau;
                vcross(tau, pi.n, pj.n);
                pi.tau.x+=J*tau.x;
                pi.tau.y+=J*tau.y;
                pi.tau.z+=J*tau.z;
            }
        }
        //put back the particle in the list
        _system.particles[pindex_i] = pi;
    }

}
