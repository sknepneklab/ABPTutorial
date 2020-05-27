#include "integrator_brownian_rotational.hpp"
#include "../box/pbc.hpp"

DEV_LAUNCHABLE
void IntegratorBrownianParticlesRotational_kernel(const int Numparticles,
                                                  ParticleType *particles,
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
    real theta = mu * dt * particles[pindex].tau;
    if (temp > 0.0)
    {
      ///< Copy state to local memory for efficiency
      curandStatePhilox4_32_10_t localState = rnd_state[pindex  ];
      ///< Generate pseudo-random
      double2 rnd_gausian = curand_normal2_double(&localState);
      theta += B * sqrt_dt * rnd_gausian.x;
      ///< Copy state back to global memory
      rnd_state[pindex] = localState;
    }
    /*
      Rotate the vector in plane.
      Parameter
      ---------
        theta : rotaton angle
    */
    real c = cos(theta);
    real s = sin(theta);
    real nx = c * particles[pindex].n.x - s * particles[pindex].n.y;
    real ny = s * particles[pindex].n.x + c * particles[pindex].n.y;
    real len = sqrt(nx * nx + ny * ny);
    // Update particle director (normalize it along the way to collect for any numerical drift that may have occurred)
    particles[pindex].n.x = nx / len;
    particles[pindex].n.y = ny / len;
  }
}

/*! Integrates equation of motion in the over-damped limit using a first order 
 *  scheme. 
 *  \note This integrator applies only to the particle position and does not implement activity.
 *  In order to use activity, you should define, e.g. external self propulsion.  
**/
void IntegratorBrownianParticlesRotational::poststep(void)
{
  IntegratorBrownianParticlesRotational_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                                          device::raw_pointer_cast(&_system.particles[0]),
                                                                                                          rng->rng_state_ptr(),
                                                                                                          this->get_temperature(),
                                                                                                          mu,
                                                                                                          B,
                                                                                                          sqrt_dt,
                                                                                                          this->get_time_step());
}