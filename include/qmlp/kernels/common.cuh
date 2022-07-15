#pragma once

#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

#define QUICKMLP_KERNEL_NAMESPACE ::qmlp::kernel
#define QUICKMLP_KERNEL_NAMESPACE_BEGIN namespace qmlp { namespace kernel {
#define QUICKMLP_KERNEL_NAMESPACE_END }}

QUICKMLP_KERNEL_NAMESPACE_BEGIN

#ifdef __CUDACC__
constexpr
#endif
__host__ __device__ inline half hZERO()
{
    return __half_raw{ 0 };
}

#ifdef __CUDACC__
constexpr
#endif
__host__ __device__ inline half2 h2ZERO()
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

struct zero_initialization_tag {};

template<typename T, int N>
class StaticArray
{
    T data_[N];

public:
    __forceinline__ __host__ __device__ StaticArray(){}
    __host__ __device__ StaticArray(zero_initialization_tag /*tag*/)
        : data_{ 0 }
    {}

    __forceinline__ __host__ __device__ constexpr const T& operator[](int i) const { return data_[i]; }
    __forceinline__ __host__ __device__ constexpr T& operator[](int i) { return data_[i]; }

    __forceinline__ __host__ __device__ StaticArray<T, N> replace(int d, T value)
    {
        StaticArray<T, N> self = *this;
        self[d] = value;
        return self;
    }
};

#if 0
template<int N>
class StaticArray<half, N>
{
    static constexpr int NDiv2 = N / 2;
    static_assert(NDiv2 * 2 == N, "N not divisible by 2");

    half2 data_[NDiv2];

public:
    __device__ StaticArray()
        : data_{ 0 }
    {}
};
#endif

QUICKMLP_KERNEL_NAMESPACE_END
