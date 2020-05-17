/************************************************************************************
* MIT License                                                                       *
*                                                                                   *
* Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           *
*                    Dr. Rastko Sknepnek, University of Dundee                      *
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
#ifndef __rng_hpp__
#define __rng_hpp__

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
typedef std::unique_ptr<RNG> RNG_ptr; 

#endif
/** @} */ 
