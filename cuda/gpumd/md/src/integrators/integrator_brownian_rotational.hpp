/************************************************************************************
* MIT License                                                                       *
*                                                                                   *
* Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           *
*               fdamatoz@gmail.com                                                  *
* Permission is hereby granted, free of charge, to any person obtaining a copy      *
* of this software and associated documentation files (the "Software"), to deal     *
* in the Software without restriction, including without limitation the rights      *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         *
* copies of the Software, and to permit persons to whom the Software is             *
* furnished to do so, subject to the following conditions:                          *
*                                                                                   *
* The above copyright notice and this permission notice shall be included in all    *
* copies or substantial portions of the Software.                                   *
*                                                                                   *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     *
* SOFTWARE.                                                                         *
*************************************************************************************/
#ifndef __integrator_brownian_rotational__hpp__
#define __integrator_brownian_rotational__hpp__

/** @addtogroup integrators Vertex Brownian Integrator
 *  @brief IntegratorBrownianParticlesRotational class
 *  @{
 */

#include <iostream>

#include "integratorclass.hpp"
#include "../rng/devrng.hpp"

/**
 * @class IntegratorBrownianParticlesRotational
 * @brief Integrator Brownian class implements Brownian dynamics for the particles position. Particle director will not be integrated
 */
class IntegratorBrownianParticlesRotational : public IntegratorClass
{
public:
  /** @brief VertexIntegrator Constructor */
  IntegratorBrownianParticlesRotational(SystemClass &system) : IntegratorClass(system)
  {
    name = "brownian";
    type = "director";
    this->set_default_properties();
  }
  /** @brief destructor */
  ~IntegratorBrownianParticlesRotational() {}

  void set_default_properties(void)
  {
    gamma = 1.0;
    mu = 1.0 / gamma;
    this->set_temperature(0.0);
    this->set_time_step(5e-3);
    seed = 123456; ///default value
    rng = std::make_unique<RNGDEV>(seed, _system.Numparticles);
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

  //using IntegratorClass::set_property;
  void set_property(const std::string &prop_name, double &value)
  {
    if (prop_name.compare("T") == 0)
    {
      this->set_temperature(value);
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
      rng = std::make_unique<RNGDEV>(seed, _system.Numparticles);
    }
    else
      this->print_warning_property_name(prop_name);
  }

  /**  @brief Propagate system for a time step */
  void prestep(void) {}

  void poststep(void);

private:
  real gamma;        //!< Friction coefficient
  real mu;           //!< Mobility (1/gamma)
  real B, sqrt_dt;   //!< useful quantities
  unsigned int seed; //!< random number seed;
  RNGDEV_ptr rng;    //!< Random number generator
};

#endif
/** @} */
