#ifndef __DEV_TYPES_HPP__
#define __DEV_TYPES_HPP__

#define __CUDA_ENABLE__ 

#if defined(__CUDA_ENABLE__)
#include "cuda_runtime.h"
//defines
#define DEV_LAUNCHABLE             __global__
#define DEV_LAMBDA                 __device__
#define DEV_INLINE_LAMBDA          __device__ inline
#define DEV_CALLABLE               __device__
#define DEV_CALLABLE_INLINE        __host__ __device__ inline
#define DEV_CALLABLE_MEMBER        __host__ __device__
#define DEV_CALLABLE_INLINE_MEMBER __host__ __device__ inline

//others
using devStream_t = cudaStream_t;

#else

//defines
#define DEV_LAUNCHABLE             
#define DEV_LAMBDA                 
#define DEV_CALLABLE               
#define DEV_CALLABLE_INLINE        inline
#define DEV_CALLABLE_MEMBER       
#define DEV_CALLABLE_INLINE_MEMBER inline

//others
using devStream_t = int;
using double2 = struct{ double x,y;};

#endif //end __CUDACC__

// Helper macro for defining device functors that can be launched as kernels
#define DEV_KERNEL_FUNCTION(name, ...)                \
  struct name {                                             \
      DEV_CALLABLE_MEMBER void operator()(__VA_ARGS__) const;  \
  };                                                        \
  DEV_CALLABLE_MEMBER void name::operator()(__VA_ARGS__) const
;


DEV_CALLABLE_INLINE
unsigned int globalThreadIndex() {
#ifdef CUDA_ENABLE
return threadIdx.x + blockIdx.x * blockDim.x;
#else
return 0;
#endif
}


DEV_CALLABLE_INLINE
unsigned int globalThreadCount() {
#ifdef CUDA_ENABLE
return blockDim.x * gridDim.x;
#else
return 1;
#endif
}


DEV_CALLABLE_INLINE
unsigned int globalBlockCount() {
#ifdef CUDA_ENABLE
return gridDim.x;
#else
return 1;
#endif
}


DEV_CALLABLE_INLINE
unsigned int localThreadIndex() {
#ifdef CUDA_ENABLE
return threadIdx.x;
#else
return 0;
#endif
}


DEV_CALLABLE_INLINE
unsigned int localThreadCount() {
#ifdef CUDA_ENABLE
return blockDim.x;
#else
return 1;
#endif
}


DEV_CALLABLE_INLINE
unsigned int globalBlockIndex() {
#ifdef CUDA_ENABLE
return blockIdx.x;
#else
return 0;
#endif
}


DEV_CALLABLE_INLINE
void synchronize() {
#ifdef CUDA_ENABLE
__syncthreads();
#endif
}
#endif // __DEV_TYPES_HPP__

