#ifndef __atomic_exch_double_HPP__
#define __atomic_exch_double_HPP__
#pragma once

#include "../configuration/dev_types.hpp"

namespace device
{
    DEV_INLINE_LAMBDA
    double atomicExch(double *address, double val)
    {
        unsigned long long int *address_as_ull = (unsigned long long int *)address;
// Make these ranges usable inside CUDA C++ device code
#ifdef __CUDA_ENABLE__
        unsigned long long res = atomicExch(address_as_ull, __double_as_longlong(val));
        return __longlong_as_double(res);
#else
        /* todo:
    modify this to work with the cpu too
    */
#endif
    }
} // namespace device

#endif