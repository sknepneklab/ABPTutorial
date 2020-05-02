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
    // Update angular velocity
    real U = _system.particles[pindex].N.x;
    real V = _system.particles[pindex].N.y;
    real W = _system.particles[pindex].N.z;
    real omega = _system.particles[pindex].tau.x * U + _system.particles[pindex].tau.y * V + _system.particles[pindex].tau.z * W;
    omega *= mu;
    // Change orientation of the director
    real theta = this->get_time_step() * omega;
    // Compute angle sins and cosines
    real c = cos(theta), s = sin(theta);
    // Rotate the director
    real nx = U * (U * _system.particles[pindex].n.x + V * _system.particles[pindex].n.y + W * _system.particles[pindex].n.z) * (1.0 - c) + _system.particles[pindex].n.x * c + (-W * _system.particles[pindex].n.y + V * _system.particles[pindex].n.z) * s;
    real ny = V * (U * _system.particles[pindex].n.x + V * _system.particles[pindex].n.y + W * _system.particles[pindex].n.z) * (1.0 - c) + _system.particles[pindex].n.y * c + (W * _system.particles[pindex].n.x - U * _system.particles[pindex].n.z) * s;
    real nz = W * (U * _system.particles[pindex].n.x + V * _system.particles[pindex].n.y + W * _system.particles[pindex].n.z) * (1.0 - c) + _system.particles[pindex].n.z * c + (-V * _system.particles[pindex].n.x + U * _system.particles[pindex].n.y) * s;
    real len = sqrt(nx * nx + ny * ny + nz * nz);

    // Update particle director (normalize it along the way to collect for any numerical drift that may have occurred)
    _system.particles[pindex].n.x = nx / len;
    _system.particles[pindex].n.y = ny / len;
    _system.particles[pindex].n.z = nz / len;
  }
}