---
layout: default
title:  "Implementing a 2D simulation of Active Brownian Particles (ABP) in CUDA"
date:   2020-05-27
---
# Harmonic Force implementation in CUDA

```c++
__global__
void HarmonicForce_kernel(const int Numparticles,
                          ParticleType *particles,
                          const int max_ng_per_particle,
                          const int * __restrict__ nglist,
                          const BoxType _box,
                          const real k,
                          const real a,
                          const real a2,
                          const bool COMPUTE_ENERGY)

{
    int pindex_i = blockIdx.x*blockDim.x + threadIdx.x;
    if(pindex_i<Numparticles)
    {
        ParticleType pi = particles[pindex_i];
        //loop over the neighbours particles
        for (int c = 0; c < pi.coordination; c++)
        {
            int pindex_j = nglist[c + max_ng_per_particle * pindex_i];
            const ParticleType pj = particles[pindex_j];
            real2 rij = device::minimum_image(pi.r, pj.r, _box);
            real lendr = vdot(rij, rij);
            if (lendr <= a2)
            {

            	real factor = -k * (a - lendr) / lendr;
            	pi.forceC.x += factor * rij.x;
            	pi.forceC.y += factor * rij.y;

            }
        }
        particles[pindex_i] = pi;
    }
}

void HarmonicForce::compute_energy(void)
{
    HarmonicForce_kernel<<<(_system.Numparticles+255)/256, 256)>>>(_system.Numparticles,
                                                                    device::raw_pointer_cast(&_system.particles[0]),
                                                                    _neighbourslist.max_ng_per_particle,
                                                                    device::raw_pointer_cast(&_neighbourslist.nglist[0]),
                                                                    _system.get_box(),
                                                                    k,
                                                                    a,
                                                                    a2);
}
```