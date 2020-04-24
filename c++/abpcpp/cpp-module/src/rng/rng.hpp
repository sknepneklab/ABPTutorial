/* ***************************************************************************
 *
 *  Copyright (C) 2017 University of Dundee
 *  All rights reserved. 
 *
 *  This file is part of AJM (Active Junction Model) program.
 *
 *  AJM is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  AJM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ****************************************************************************/

/*!
 * \file rng.hpp
 * \author Rastko Sknepnek, sknepnek@gmail.com
 * \date 24-Oct-2013
 * \brief Class RNG provides wrappers for the GSL random number generate
 */ 

#ifndef __RNG_HPP__
#define __RNG_HPP__

/** @addgroup utilities Random Number Generator 
 *  @brief Random Number Generator Class 
 *  @{
 */

#include <random>
#include <memory>

/**
 * @class RNG 
 * @brief Class handles random numbers in the system 
 * 
 */
class RNG
{
public:
  
  //! Constructor (initialize random number generator)
  RNG(unsigned int seed) : _generator(seed), _uniform_distribution(0.0,1.0), _normal_distribution(0.0,1.0) { }
  
  //! Destructor
  ~RNG() { }
  
  //! Get a random number between 0 and 1 drawn from an uniform distribution
  //! \return random number between 0 and 1
  double drnd()
  {
    return _uniform_distribution(_generator);
  }

  //! Return a random number from a Gaussian distribution with a given standard deviation 
  //! \param sigma standard deviation 
  double gauss_rng(double sigma, double mu = 0.0)
  {
    return (_normal_distribution(_generator)*sigma + mu);
  }

  //! Get an integer random number between 0 and N drawn from an uniform distribution
  //! \param N upper bound for the interval
  //! \return integer random number between 0 and N
  int lrnd(int N)
  {
    return static_cast<int>(N*drnd());
  }

private:
  
  std::mt19937_64 _generator;  //!< Mersenne Twister engine 
  std::uniform_real_distribution<double> _uniform_distribution;  // Uniform random numbers
  std::normal_distribution<double> _normal_distribution; // Gaussian distribution zero mean, unit variance
  
};

/**
 * @brief unique pointer to the RNG class
 */
typedef std::shared_ptr<RNG> RNG_ptr; 

#endif
/** @} */ 
