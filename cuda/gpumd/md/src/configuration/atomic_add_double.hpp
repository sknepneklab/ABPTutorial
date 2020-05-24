#ifndef __ATOMIC_ADD_DOUBLE_HPP__
#define __ATOMIC_ADD_DOUBLE_HPP__
#pragma once

#include "../configuration/dev_types.hpp"

/* todo:
modify this to work with the cpu too
*/
namespace device
{

    DEV_INLINE_LAMBDA
    double double_atomicAdd(double *address, double val)
    {
        unsigned long long int *address_as_ull = (unsigned long long int *)address;
        unsigned long long int old = *address_as_ull, assumed;
// Make these ranges usable inside CUDA C++ device code
#ifdef __CUDA_ENABLE__
        do
        {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                                 __longlong_as_double(assumed)));

            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
#else
        /* todo:
    modify this to work with the cpu too
    */
#endif
    }
} // namespace device

#endif