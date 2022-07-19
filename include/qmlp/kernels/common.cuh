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


/**
 * Templated casting between float and half datatypes
 */
template<typename Dst, typename Src>
__forceinline__ __host__ __device__ Dst fcast(const Src& src);

template<>
__forceinline__ __host__ __device__ float fcast<float, half>(const half& src) { return __half2float(src); }
template<>
__forceinline__ __host__ __device__ float fcast<float, float>(const float& src) { return src; }
template<>
__forceinline__ __host__ __device__ half fcast<half, half>(const half& src) { return src; }
template<>
__forceinline__ __host__ __device__ half fcast<half, float>(const float& src) { return __float2half(src); }

struct zero_initialization_tag {};

template<typename T, int N>
class StaticArray
{
    T data_[N];

public:
    constexpr __forceinline__ __host__ __device__ StaticArray(){}
    constexpr __host__ __device__ StaticArray(zero_initialization_tag /*tag*/)
        : data_{ 0 }
    {}
    constexpr __host__ __device__ StaticArray(T initialValue)
    {
        for (int i = 0; i < N; ++i) data_[i] = initialValue;
    }

    constexpr __forceinline__ __host__ __device__ const T& operator[](int i) const { return data_[i]; }
    constexpr __forceinline__ __host__ __device__ T& operator[](int i) { return data_[i]; }

    constexpr __forceinline__ __host__ __device__ StaticArray<T, N> replace(int d, T value) const
    {
        StaticArray<T, N> self = *this;
        self[d] = value;
        return self;
    }

    /**
     * Multiplies all array elements together, but ignores dimension 'd'.
     */
    constexpr __forceinline__ __host__ __device__ T reduceMulWithoutD(int d, int start=0) const
    {
        T result = T(1);
        for (int i = start; i < N; ++i)
            if (i != d) result *= data_[i];
        return result;
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
