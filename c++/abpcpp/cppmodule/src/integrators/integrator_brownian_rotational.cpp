#include "integrator_brownian_rotational.hpp"
#include "../box/pbc.hpp"

/*! Integrates equation of motion in the over-damped limit using a first order 
 *  scheme. 
 *  \note This integrator applies only to the particle position and does not implement activity.
 *  In order to use activity, you should define, e.g. external self propulsion.  
**/
void IntegratorBrownianParticlesRotational::poststep(void)
{
  for (int pindex = 0; pindex < _system.Numparticles; pindex++)
  {
    real theta = (this->get_time_step()/mu)*_system.particles[pindex].tau;
    if (this->get_temperature() > 0.0)
    {
      theta+=B * rng->gauss_rng(0.0,1.0);
    }
    /*
      Rotate the vector in plane.
      Parameter
      ---------
        theta : rotaton angle
    */
    real c = cos(theta);
    real s = sin(theta);
    real nx = c*_system.particles[pindex].n.x - s*_system.particles[pindex].n.y;
    real ny = s*_system.particles[pindex].n.x + c*_system.particles[pindex].n.y; 
    real len = sqrt(nx * nx + ny * ny);
    // Update particle director (normalize it along the way to collect for any numerical drift that may have occurred)
    _system.particles[pindex].n.x = nx / len;
    _system.particles[pindex].n.y = ny / len;
  }
}