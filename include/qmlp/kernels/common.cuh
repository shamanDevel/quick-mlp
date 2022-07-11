#pragma once

#ifndef CUDA_NO_HOST
#include <host_defines.h>
#endif

#define QUICKMLP_KERNEL_NAMESPACE ::qmlp::kernel
#define QUICKMLP_KERNEL_NAMESPACE_BEGIN namespace qmlp { namespace kernel {
#define QUICKMLP_KERNEL_NAMESPACE_END }}

QUICKMLP_KERNEL_NAMESPACE_BEGIN

constexpr __host__ __device__ half hZERO()
{
    return __half_raw{ 0 };
}

constexpr __host__ __device__ half2 h2ZERO()
{
    return __half2_raw{ 0, 0 };
}

template<typename A, typename B>
constexpr __host__ __device__ auto max(A a, B b)
{
    return a > b ? a : b;
}

template<typename A, typename B>
constexpr __host__ __device__ auto min(A a, B b)
{
    return a < b ? a : b;
}

QUICKMLP_KERNEL_NAMESPACE_END
