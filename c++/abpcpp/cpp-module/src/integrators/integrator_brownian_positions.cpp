#include "integrator_brownian_positions.hpp"
#include "../box/pbc.hpp"


/*! Integrates equation of motion in the over-damped limit using a first order 
 *  scheme. 
 *  \note This integrator applies only to the particle position and does not implement activity.
 *  In order to use activity, you should define, e.g. external self propulsion.  
**/
void IntegratorBrownianParticlesPositions::poststep(void)
{
  for (int pindex = 0; pindex < _system.Numparticles; pindex ++)
  {
    real3 force_rnd;
    force_rnd.x = force_rnd.y = force_rnd.z = 0.0;
    if (this->get_temperature() > 0.0)
    {
      force_rnd.x = B * rng->gauss_rng(0.0,1);
      force_rnd.y = B *rng->gauss_rng(0.0,1);
      force_rnd.z = B *rng->gauss_rng(0.0,1);
    }
    // Update particle position
    _system.particles[pindex].r.x += mu * this->get_time_step() * _system.particles[pindex].forceC.x + sqrt_dt * force_rnd.x;
    _system.particles[pindex].r.y += mu * this->get_time_step() * _system.particles[pindex].forceC.y + sqrt_dt * force_rnd.y;
    _system.particles[pindex].r.z += mu * this->get_time_step() * _system.particles[pindex].forceC.z + sqrt_dt * force_rnd.z;

    //apply boundary conditions
    host::enforce_periodic(_system.particles[pindex].r, _system.particles[pindex].ip, _system.get_box());
  }
}