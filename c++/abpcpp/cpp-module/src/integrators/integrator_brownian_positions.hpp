#ifndef __integrator_brownian_positions__hpp__
#define __integrator_brownian_positions__hpp__

/** @addtogroup integrators Vertex Brownian Integrator
 *  @brief IntegratorBrownianParticlesPositions class
 *  @{
 */

#include <iostream>

#include "integratorclass.hpp"
#include "../rng/rng.hpp"

/**
 * @class IntegratorBrownianParticlesPositions
 * @brief Integrator Brownian class implements Brownian dynamics for the particles position. Particle director will not be integrated
 */
class IntegratorBrownianParticlesPositions : public IntegratorClass
{
public:
  /** @brief VertexIntegrator Constructor */
  IntegratorBrownianParticlesPositions(SystemClass &system) : IntegratorClass(system)
  {
    name = "brownian";
    type = "positions";
    this->set_defaults_property();
  }
  /** @brief destructor */
  ~IntegratorBrownianParticlesPositions() {}

  void set_defaults_property(void)
  {
    gamma = 1.0;
    mu = 1.0 / gamma;
    this->set_temperature(0.0);
    this->set_time_step(5e-3);
    seed = 123456; ///default value
    rng = std::make_shared<RNG>(seed);
  }

  /** @brief Update the temperature dependent parameters **/
  void update_temperature_parameters()
  {
    B = sqrt(2.0 * this->get_temperature() * mu);
  }
  /** @brief Update the temperature dependent parameters **/
  void update_time_step_parameters()
  {
    sqrt_dt = sqrt(this->get_time_step());
  }

  using IntegratorClass::set_property;
  void set_property(const std::string &prop_name, double &value)
  {
    if (prop_name.compare("T") == 0)
    {
      this->set_temperature(T);
      this->update_temperature_parameters();
    }
    else if (prop_name.compare("gamma") == 0)
    {
      gamma = value;
      mu = 1.0 / gamma;
      this->update_temperature_parameters();
    }
    else if (prop_name.compare("seed") == 0)
    {
      seed = uint(value);
      rng = std::make_shared<RNG>(seed);
    }
    else
      print_warning_property_name(prop_name);
  }

  /**  @brief Propagate system for a time step */
  void prestep(void) {}

  void poststep(void);

private:
  real gamma;        //!< Friction coefficient
  real mu;           //!< Mobility (1/gamma)
  real B, sqrt_dt;   //!< useful quantities
  unsigned int seed; //!< random number seed;
  RNG_ptr rng;       //!< Random number generator
};

#endif
/** @} */
