#include "../box/pbc_device.hpp"
#include "harmonicforce.hpp"

DEV_LAUNCHABLE
void HarmonicForce_kernel(const int Numparticles,
                          ParticleType *particles,
                          const int max_ng_per_particle,
                          const int *__restrict__ nglist,
                          const BoxType _box,
                          const real k,
                          const real a,
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
            real lendr = sqrt(vdot(rij, rij));
            //real lendr = norm3d(rij.x, rij.y, 0.0);
            if (lendr <= a)
            {
                if (!COMPUTE_ENERGY)
                {
                    real factor = -k * (a - lendr) / lendr;
                    pi.forceC.x += factor * rij.x;
                    pi.forceC.y += factor * rij.y;
                }
                else
                    pi.energy += 0.5 * k * (a - lendr) * (a - lendr);
            }
        }
        particles[pindex_i] = pi;
    }
}

void HarmonicForce::compute_energy(void)
{
    HarmonicForce_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                    device::raw_pointer_cast(&_system.particles[0]),
                                                                                    _neighbourslist.max_ng_per_particle,
                                                                                    device::raw_pointer_cast(&_neighbourslist.nglist[0]),
                                                                                    _system.get_box(),
                                                                                    k,
                                                                                    a,
                                                                                    true);
}

void HarmonicForce::compute(void)
{
    HarmonicForce_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                    device::raw_pointer_cast(&_system.particles[0]),
                                                                                    _neighbourslist.max_ng_per_particle,
                                                                                    device::raw_pointer_cast(&_neighbourslist.nglist[0]),
                                                                                    _system.get_box(),
                                                                                    k,
                                                                                    a,
                                                                                    false);
}
