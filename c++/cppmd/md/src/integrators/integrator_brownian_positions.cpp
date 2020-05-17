#include "integrator_brownian_positions.hpp"
#include "../box/pbc.hpp"

/*! Integrates equation of motion in the over-damped limit using a first order 
 *  scheme. 
 *  \note This integrator applies only to the particle position and does not implement activity.
 *  In order to use activity, you should define, e.g. external self propulsion.  
**/
void IntegratorBrownianParticlesPositions::poststep(void)
{
  for (int pindex = 0; pindex < _system.Numparticles; pindex++)
  {
    real2 force_rnd;
    force_rnd.x = force_rnd.y = 0.0;
    if (this->get_temperature() > 0.0)
    {
      force_rnd.x = B * sqrt_dt * rng->gauss_rng(1.0, 0.0);
      force_rnd.y = B * sqrt_dt * rng->gauss_rng(1.0, 0.0);
    }
    // Update particle position
    _system.particles[pindex].r.x += mu * this->get_time_step() * _system.particles[pindex].forceC.x + sqrt_dt * force_rnd.x;
    _system.particles[pindex].r.y += mu * this->get_time_step() * _system.particles[pindex].forceC.y + sqrt_dt * force_rnd.y;

    //apply boundary conditions
    host::enforce_periodic(_system.particles[pindex].r, _system.particles[pindex].ip, _system.get_box());
  }
}