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



#endif // __DEV_TYPES_HPP__

