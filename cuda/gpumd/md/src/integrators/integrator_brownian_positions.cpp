#include "integrator_brownian_positions.hpp"
#include "../box/pbc_device.hpp"

DEV_LAUNCHABLE
void IntegratorBrownianParticlesPositions_kernel(const int Numparticles,
                                                 ParticleType *particles,
                                                 const BoxType _box,
                                                 curandStatePhilox4_32_10_t *rnd_state,
                                                 const real temp,
                                                 const real mu,
                                                 const real B,
                                                 const real sqrt_dt,
                                                 const real dt)

{
  for (int pindex = blockIdx.x * blockDim.x + threadIdx.x;
       pindex < Numparticles;
       pindex += blockDim.x * gridDim.x)
  {
    real2 force_rnd;
    force_rnd.x = force_rnd.y = 0.0;
    if (temp > 0.0)
    {
      ///< Copy state to local memory for efficiency
      curandStatePhilox4_32_10_t localState = rnd_state[pindex];
      ///< Generate pseudo-random
      double2 rnd_gausian = curand_normal2_double(&localState);
      force_rnd.x = B * rnd_gausian.x;
      force_rnd.y = B * rnd_gausian.y;
      ///< Copy state back to global memory
      rnd_state[pindex] = localState;
    }
    // Update particle position
    particles[pindex].r.x += mu * dt * particles[pindex].forceC.x + sqrt_dt * force_rnd.x;
    particles[pindex].r.y += mu * dt * particles[pindex].forceC.y + sqrt_dt * force_rnd.y;

    //apply boundary conditions
    device::enforce_periodic(particles[pindex].r, particles[pindex].ip, _box);
  }
}

/*! Integrates equation of motion in the over-damped limit using a first order 
 *  scheme. 
 *  \note This integrator applies only to the particle position and does not implement activity.
 *  In order to use activity, you should define, e.g. external self propulsion.  
**/
void IntegratorBrownianParticlesPositions::poststep(void)
{
  IntegratorBrownianParticlesPositions_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                                         device::raw_pointer_cast(&_system.particles[0]),
                                                                                                         _system.get_box(),
                                                                                                         rng->rng_state_ptr(),
                                                                                                         this->get_temperature(),
                                                                                                         mu,
                                                                                                         B,
                                                                                                         sqrt_dt,
                                                                                                         this->get_time_step());
}
