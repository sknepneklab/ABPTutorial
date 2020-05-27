---
layout: default
title:  "Implementing a 2D simulation of Active Brownian Particles (ABP) in C++"
date:   2020-05-27
---
# Harmonic Force implementation in C++

```c++
void HarmonicForce::compute(void)
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
            real lendr = sqrt(vdot(rij, rij));
            if (lendr <= a)
            {
                real factor = -k*(a - lendr)/lendr;
                pi.forceC.x+=factor*rij.x;
                pi.forceC.y+=factor*rij.y;
            }
        }
        //put back the particle in the list
        _system.particles[pindex_i] = pi;
    }
    /**/
}
```