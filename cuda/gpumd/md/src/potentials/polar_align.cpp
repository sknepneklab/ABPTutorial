#include "../box/pbc_device.hpp"
#include "polar_align.hpp"

DEV_LAUNCHABLE
void PolarAlign_kernel(const int Numparticles,
                       ParticleType *particles,
                       const int max_ng_per_particle,
                       const int *__restrict__ nglist,
                       const BoxType _box,
                       const real J,
                       const real a2,
                       const bool COMPUTE_ENERGY)

{
    for (int pindex_i = blockIdx.x * blockDim.x + threadIdx.x;
         pindex_i < Numparticles;
         pindex_i += blockDim.x * gridDim.x)
    {
        ParticleType pi = particles[pindex_i];
        //loop over the neighbours particles
        for (int c = 0; c < pi.coordination; c++)
        {
            int pindex_j = nglist[c + max_ng_per_particle * pindex_i];
            const ParticleType pj = particles[pindex_j];
            real2 rij = device::minimum_image(pi.r, pj.r, _box);
            real lendr2 = vdot(rij, rij);
            if (lendr2 <= a2)
            {
                if (!COMPUTE_ENERGY)
                {
                    pi.tau += J * vcross(pi.n, pj.n);
                }
                else
                    pi.energy += J * vdot(pi.n, pj.n);
            }
        }
        particles[pindex_i] = pi;
    }
}

void PolarAlign::compute_energy(void)
{
    PolarAlign_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                 device::raw_pointer_cast(&_system.particles[0]),
                                                                                 _neighbourslist.max_ng_per_particle,
                                                                                 device::raw_pointer_cast(&_neighbourslist.nglist[0]),
                                                                                 _system.get_box(),
                                                                                 J,
                                                                                 a2,
                                                                                 true);
}

void PolarAlign::compute(void)
{
    
    PolarAlign_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                 device::raw_pointer_cast(&_system.particles[0]),
                                                                                 _neighbourslist.max_ng_per_particle,
                                                                                 device::raw_pointer_cast(&_neighbourslist.nglist[0]),
                                                                                 _system.get_box(),
                                                                                 J,
                                                                                 a2,
                                                                                 false);
}
