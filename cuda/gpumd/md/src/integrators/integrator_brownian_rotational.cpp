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
  for (int pindex = globalThreadIndex();
       pindex < Numparticles;
       pindex += globalThreadCount())
  {
    real theta = mu * dt * particles[pindex].tau;
    if (temp > 0.0)
    {
      ///< Copy state to local memory for efficiency
      curandStatePhilox4_32_10_t localState = rnd_state[pindex  ];
      ///< Generate pseudo-random
      double2 rnd_gausian = curand_normal2_double(&localState);
      theta += B * sqrt_dt * rnd_gausian.x;
    }
    /*
      Rotate the vector in plane.
      Parameter
      ---------
        theta : rotaton angle
    */
    float c = cosf(theta);
    float s = sinf(theta);
    float nx = c * particles[pindex].n.x - s * particles[pindex].n.y;
    float ny = s * particles[pindex].n.x + c * particles[pindex].n.y;
    float len = sqrtf(nx * nx + ny * ny);
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