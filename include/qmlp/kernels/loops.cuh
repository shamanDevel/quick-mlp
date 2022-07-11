#pragma once

#ifndef CUDA_NO_HOST
#include <host_defines.h>
#endif

#include <qmlp/kernels/common.cuh>

QUICKMLP_KERNEL_NAMESPACE_BEGIN

/**
 * Computes integer division x/y with round-up mode.
 * All values must be non-negative (and y positive)!
 */
template<typename T>
__forceinline__ __host__ __device__ T divRoundUp(T x, int y)
{
	//https://stackoverflow.com/a/2745086/1786598
	return (x + y - 1) / y;
}

template<typename T>
__forceinline__ __host__ __device__ T roundUp(T numToRound, int multiple)
{
	//assert(multiple);
	return ((numToRound + multiple - 1) / multiple) * multiple;
}

/**
 * Rounds up the non-negative number 'numToRound'
 * up to the next multiple of the positive number 'multiple'.
 * 'multiple' must be a power of 2.
 */
template<typename T>
__forceinline__ __host__ __device__ T roundUpPower2(T numToRound, int multiple)
{
	//assert(multiple && ((multiple & (multiple - 1)) == 0));
	return (numToRound + multiple - 1) & -multiple;
}

#define KERNEL_3D_LOOP(i, j, k, virtual_size) 												\
	for (ptrdiff_t __i = blockIdx.x * blockDim.x + threadIdx.x;							\
		 __i < virtual_size.x*virtual_size.y*virtual_size.z;										\
		 __i += blockDim.x * gridDim.x) {															\
		 ptrdiff_t k = __i / (virtual_size.x*virtual_size.y);							\
		 ptrdiff_t j = (__i - (k * virtual_size.x*virtual_size.y)) / virtual_size.x;	\
		 ptrdiff_t i = __i - virtual_size.x * (j + virtual_size.y * k);
#define KERNEL_3D_LOOP_END }

#define KERNEL_1D_LOOP(i, numel)								\
	for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;	\
		 i < numel;												\
		 i += blockDim.x * gridDim.x) {
#define KERNEL_1D_LOOP_END }

#define KERNEL_1D_LOOP_SYNC(i, valid, numel)							\
    const auto numel_pow32 = qmlp::kernel::roundUpPower2(numel, 32);	\
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;			\
		 i < numel_pow32;												\
		 i += blockDim.x * gridDim.x) {									\
		 const bool valid = i < numel;

#define KERNEL_1D_LOOP_BLOCKSYNC(i, valid, numel)						\
    const auto numel_pow32 = qmlp::kernel::roundUpPower2(numel, 32);	\
    for (ptrdiff_t i = threadIdx.x;										\
		 i < numel_pow32;												\
		 i += blockDim.x) {												\
		 const bool valid = i < numel;

QUICKMLP_KERNEL_NAMESPACE_END
