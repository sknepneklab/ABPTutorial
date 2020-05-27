#include "../configuration/dev_types.hpp"
#include "devrng.hpp"

DEV_LAUNCHABLE 
void rng_setup_kernel(unsigned int seed_dev, 
                      curandStatePhilox4_32_10_t *rng_state, 
                      const unsigned int Nrng)
{
  //const int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //if(tid < Nrng)
  ///< Each thread gets same seed, a different sequence number, no offset
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < Nrng;
       tid += blockDim.x * gridDim.x)
  {
    curand_init(seed_dev, tid, 0, &rng_state[tid]);
  }
}

void RNGDEV::rng_setup_device(unsigned int seed_dev, const unsigned int Nrng)
{
  //std::cout<< "\n***********************************************\n";
  //std::cout<< "****\tRNG SETUP DEVICE\t****\n";
  //std::cout<< "***********************************************\n\n";
  //std::cout<< "Philox4_32_10_t\n";
  rng_setup_kernel<<<(Nrng + 255) / 256, 256>>>(seed_dev, rng_state, Nrng);
  cudaDeviceSynchronize();
}

void RNGDEV::rng_delete(void)
{
  cudaFree(rng_state);
  rng_isAlloced = false;
}