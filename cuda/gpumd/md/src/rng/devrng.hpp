/**
 * @brief GPU random number generator class.
 * @file devRND.hpp
 * @author D.A Matoz-Fernandez
 * @date 2018-05-24
 */

#ifndef __DEVRNG_HPP__
#define __DEVRNG_HPP__

/// host
#include <memory>
#include <iostream>

/// device 
#include <curand_kernel.h>

/**
 * @brief single presicion device random number
 * @param _rng_state 
 * @return __device__ philox_4single_01 
 */
#define philox_4single_01(rnd, _rng_state)                  \
  ///< Copy state to local memory for efficiency            \
  curandStatePhilox4_32_10_t localState = _rng_state;       \
  ///< Generate pseudo-random uniforms                      \
  rnd = curand_uniform4(&_rng_state);                       \
  ///< Copy state back to global memory                     \
  _rng_state = localState;          

/**
 * @brief single presicion device random number
 * @param _rng_state 
 * @return __device__ philox_4single_01 
 */
#define philox_single_01(rng, _rng_state)                   \
  ///< Copy state to local memory for efficiency            \
  curandStatePhilox4_32_10_t localState = _rng_state;       \
  ///< Generate pseudo-random uniforms                      \
  rnd = curand_uniform(&_rng_state);                        \
  ///< Copy state back to global memory                     \
  _rng_state = localState;


/**
 * @brief double presicion device random number
 * @param _rng_state 
 * @return __device__ philox_4single_01 
 */
#define philox_2double_01(rng, _rng_state)                  \
  ///< Copy state to local memory for efficiency            \
  curandStatePhilox4_32_10_t localState = _rng_state;       \
  ///< Generate pseudo-random uniforms                      \
  rnd = curand_uniform2_double(&_rng_state);                \
  ///< Copy state back to global memory                     \
  _rng_state = localState;


/**
 * @brief double presicion device random number
 * @param _rng_state 
 * @return __device__ philox_4single_01 
 */
#define philox_double_01(rng, _rng_state)                   \
  ///< Copy state to local memory for efficiency            \
  curandStatePhilox4_32_10_t localState = _rng_state;       \
  ///< Generate pseudo-random uniforms                      \
  rnd = curand_uniform_double(&_rng_state);                 \
  ///< Copy state back to global memory                     \
  _rng_state = localState;


/**
 * @brief philox guassian random number 
 * 
 * @param _rng_state 
 * @return __device__ philox_4single_normal 
 */
#define philox_4single_normal(rng, _rng_state)              \
  ///< Copy state to local memory for efficiency            \
  curandStatePhilox4_32_10_t localState = _rng_state;       \
  ///< Generate pseudo-random uniforms                      \
  rnd = curand_normal4(&_rng_state);                        \
  ///< Copy state back to global memory                     \
  _rng_state = localState;
/**
 * @brief philox guassian random number
 * 
 * @param _rng_state 
 * @return __device__ philox_2double_normal 
 */
#define philox_2double_normal(rng, _rng_state)              \
  ///< Copygpu_configuration state to local memory for efficiency            \
  curandStatePhilox4_32_10_t localState = _rng_state;       \
  ///< Generate pseudo-random uniforms                      \
  rnd = curand_normal2_double(&localState);                 \
  ///< Copy state back to global memory                     \
  _rng_state = localState;

/**
 * @class RNGDEV
 * @brief GPU randon number genrator
 */
class RNGDEV
{
public:
  
    //! Constructor (initialize random number generator)
    RNGDEV(unsigned int seed, unsigned int size)
    { 
        cudaMalloc((void **)&rng_state, size * sizeof(curandStatePhilox4_32_10_t));
        rng_isAlloced = true;
        rng_size = size;
        rng_setup_device(seed, size);
    }

    //! Destructor
    ~RNGDEV() { cudaFree(rng_state); }
    /**
     * @brief Inititalisation
     * @param seed_dev 
     * @param Nrng 
     */
    void rng_setup_device(unsigned int seed_dev, const unsigned int Nrng);
    /**
     * @brief Delete the random generator
     */
    void rng_delete(void);
    /**
     * @brief return the state of the random
     * @return curandStatePhilox4_32_10_t* 
     */
    curandStatePhilox4_32_10_t* rng_state_ptr(){ return rng_state;}

private:
    curandStatePhilox4_32_10_t* rng_state;  //!< Random number generator state
    bool rng_isAlloced = false;             //!< Random number generator flag
    unsigned int rng_size;                  //!< Random number size
    unsigned int seed;                      //!< Random number seed 
};

/**
 * @brief unique pointer to the RNGDEV class
 */
typedef std::unique_ptr<RNGDEV> RNGDEV_ptr; //!<make_unique is an upcoming C++14 feature


#endif