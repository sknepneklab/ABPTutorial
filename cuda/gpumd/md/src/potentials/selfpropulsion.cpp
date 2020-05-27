#include "selfpropulsion.hpp"

DEV_LAUNCHABLE
void SelfPropulsionForce_kernel(const int Numparticles,
                                ParticleType *particles,
                                const real alpha)
{
    for (int pindex_i = blockIdx.x * blockDim.x + threadIdx.x;
         pindex_i < Numparticles;
         pindex_i += blockDim.x * gridDim.x)  
    {
        ParticleType pi = particles[pindex_i];
        pi.forceC.x+=alpha*pi.n.x;
        pi.forceC.y+=alpha*pi.n.y;
        //put back the particle in the list
        particles[pindex_i] = pi;
    }
}


void SelfPropulsionForce::compute(void)
{
    SelfPropulsionForce_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                    device::raw_pointer_cast(&_system.particles[0]),
                                                                                    alpha);
}
