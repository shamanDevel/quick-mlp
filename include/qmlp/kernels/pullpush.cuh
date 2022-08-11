#pragma once

/*
 * This is the template for the fully fused network.
 * Modified by the host code before compiled,
 * everything in $$..$$ will be replaced
 */

#ifndef CUDA_NO_HOST
#include "assert.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#endif
#include <cuda_fp16.h>

#include <qmlp/kernels/common.cuh>
#include <qmlp/kernels/tensor.cuh>
#include <qmlp/kernels/loops.cuh>

QUICKMLP_KERNEL_NAMESPACE_BEGIN

#define MAX_CHANNELS 16

__device__ inline int start_index(int a, int b, int c) {
    return (int)floor((float)(a * c) / b);
}

__device__ inline int end_index(int a, int b, int c) {
    return (int)ceil((float)((a + 1) * c) / b);
}

template<typename scalar_t>
__global__ void FastInpaintingFractionalKernel_Down(dim3 virtual_size,
	const Tensor3Read<scalar_t> mask,
	const Tensor4Read<scalar_t> data,
	Tensor3RW<scalar_t> maskLow,
	Tensor4RW<scalar_t> dataLow)
{
	const int H = mask.size(1);
	const int W = mask.size(2);
	const int oH = H / 2;
	const int oW = W / 2;
	const int C = data.size(1);
	KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
	{
		int Count = 0;
		scalar_t N1 = 0;
		scalar_t N2 = 0;
		scalar_t d[MAX_CHANNELS] = { 0 };
		for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
			for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
			{
				Count++;
				N1 += mask[b][ii][jj];
				N2 = max(N2, mask[b][ii][jj]);
				for (int c = 0; c < C; ++c)
					d[c] += mask[b][ii][jj] * data[b][c][ii][jj];
			}
		//maskLow[b][i][j] = N1 / Count;
		maskLow[b][i][j] = N2;
		for (int c = 0; c < C; ++c)
			dataLow[b][c][i][j] = N1 > 0 ? d[c] / N1 : 0;
	}
	KERNEL_3D_LOOP_END
}

template<typename scalar_t>
__global__ void FastInpaintingFractionalKernel_Up(dim3 virtual_size,
	const Tensor3Read<scalar_t> mask,
	const Tensor4Read<scalar_t> data,
	const Tensor3Read<scalar_t> maskLow,
	const Tensor4Read<scalar_t> dataLow,
	Tensor3RW<scalar_t> maskHigh,
	Tensor4RW<scalar_t> dataHigh)
{
	const int H = mask.size(1);
	const int W = mask.size(2);
	const int oH = H / 2;
	const int oW = W / 2;
	const int C = data.size(1);
	KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of high resolution
	{
		//interpolate from low resolution (bilinear)
		//get neighbor offsets
		int io = i % 2 == 0 ? -1 : +1;
		int jo = j % 2 == 0 ? -1 : +1;
		//accumulates
		scalar_t Weight = 0;
		scalar_t N = 0;
		scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)														\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW) {								\
		Weight += w;															\
		N += w * maskLow[b][(ii)][(jj)];										\
		for (int c = 0; c < C; ++c)												\
			d[c] += w * maskLow[b][(ii)][(jj)] * dataLow[b][c][(ii)][(jj)];		\
	}
		ITEM(i / 2, j / 2, 0.75f * 0.75f);
		ITEM(i / 2 + io, j / 2, 0.25f * 0.75f);
		ITEM(i / 2, j / 2 + jo, 0.25f * 0.75f);
		ITEM(i / 2 + io, j / 2 + jo, 0.25f * 0.25f);
#undef ITEM
		//write output
		scalar_t m = mask[b][i][j];
		maskHigh[b][i][j] = m + (N > 0 ? (1 - m) * (N / Weight) : 0);
		for (int c = 0; c < C; ++c)
		{
			dataHigh[b][c][i][j] =
				m * data[b][c][i][j] +
				(1 - m) * (N > 0 ? d[c] / N : 0);
		}
	}
	KERNEL_3D_LOOP_END
}

#define IDX4(tensor, b,c,y,x) ((b)*tensor.stride(0)+(c)*tensor.stride(1)+(y)*tensor.stride(2)+(x)*tensor.stride(3))
#define IDX3(tensor, b,y,x) ((b)*tensor.stride(0)+(y)*tensor.stride(1)+(x)*tensor.stride(2))

template<typename scalar_t>
__global__ void AdjFastInpaintingFractionKernel_Down(dim3 virtual_size,
	const Tensor3Read<scalar_t> maskIn,
	const Tensor4Read<scalar_t> dataIn,
	const Tensor3Read<scalar_t> gradMaskLowIn,
	const Tensor4Read<scalar_t> gradDataLowIn,
	Tensor3RW<scalar_t> gradMaskOut,
	Tensor4RW<scalar_t> gradDataOut)
{
	const int H = maskIn.size(1);
	const int W = maskIn.size(2);
	const int oH = H / 2;
	const int oW = W / 2;
	const int C = dataIn.size(1);
	KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
	{
		//forward
		int Count = 0;
		scalar_t N1 = 0;
		scalar_t N2 = 0;
		scalar_t d[MAX_CHANNELS] = { 0 };
		for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
			for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
			{
				Count++;
				N1 += maskIn[b][ii][jj];
				N2 = max(N2, maskIn[b][ii][jj]);
				for (int c = 0; c < C; ++c)
					d[c] += maskIn[b][ii][jj] * dataIn[b][c][ii][jj];
			}
		//maskLow[b][i][j] = N2;
		////maskLow[b][i][j] = N1 / Count;
		//for (int c = 0; c < C; ++c)
		//	dataLow[b][c][i][j] = N1 > 0 ? d[c] / N1 : 0;

		//adjoint
		//Note: no atomics since every high-res pixel is accessed only once
		scalar_t adjD[MAX_CHANNELS] = { 0 };
		scalar_t adjN1 = 0;
		for (int c = 0; c < C; ++c)
		{
			adjD[c] = N1 > 0 ? gradDataLowIn[b][c][i][j] / N1 : 0;
			adjN1 -= N1 > 0 ? gradDataLowIn[b][c][i][j] * d[c] / (N1 * N1) : 0;
		}
		scalar_t adjN2 = gradMaskLowIn[b][i][j];
		for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
			for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
			{
				scalar_t adjMask = 0;
				for (int c = 0; c < C; ++c)
				{
					gradDataOut[b][c][ii][jj] += adjD[c] * maskIn[b][ii][jj];
					adjMask += adjD[c] * dataIn[b][c][ii][jj];
				}
				adjMask += adjN1;
				//N2 = max(N2, maskIn[b][ii][jj]);
				if (N2 == maskIn[b][ii][jj])
					adjMask += adjN2;

				gradMaskOut[b][ii][jj] += adjMask;
			}
	}
	KERNEL_3D_LOOP_END
}

template<typename scalar_t>
__global__ void AdjFastInpaintingFractionalKernel_Up(dim3 virtual_size,
	const Tensor3Read<scalar_t> maskIn,
	const Tensor4Read<scalar_t> dataIn,
	const Tensor3Read<scalar_t> maskLowIn,
	const Tensor4Read<scalar_t> dataLowIn,
	const Tensor3Read<scalar_t> gradMaskHighIn,
	const Tensor4Read<scalar_t> gradDataHighIn,
	Tensor3RW<scalar_t> gradMaskOut,
	Tensor4RW<scalar_t> gradDataOut,
	Tensor3RW<scalar_t> gradMaskLowOut,
	Tensor4RW<scalar_t> gradDataLowOut)
{
	const int H = maskIn.size(1);
	const int W = maskIn.size(2);
	const int oH = H / 2;
	const int oW = W / 2;
	const int C = dataIn.size(1);
	KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of high resolution
	{
		//FORWARD

		//interpolate from low resolution (bilinear)
		//get neighbor offsets
		int io = i % 2 == 0 ? -1 : +1;
		int jo = j % 2 == 0 ? -1 : +1;
		//accumulates
		scalar_t Weight = 0;
		scalar_t N = 0;
		scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)															\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW) {								\
		Weight += w;															\
		N += w * maskLowIn[b][(ii)][(jj)];										\
		for (int c = 0; c < C; ++c)												\
			d[c] += w * maskLowIn[b][(ii)][(jj)] * dataLowIn[b][c][(ii)][(jj)];	\
	}
		ITEM(i / 2, j / 2, 0.75f * 0.75f);
		ITEM(i / 2 + io, j / 2, 0.25f * 0.75f);
		ITEM(i / 2, j / 2 + jo, 0.25f * 0.75f);
		ITEM(i / 2 + io, j / 2 + jo, 0.25f * 0.25f);
#undef ITEM
		//write output
		scalar_t m = maskIn[b][i][j];
		//maskHigh[b][i][j] = m + (1 - m) * (N / Weight);
		//for (int c = 0; c < C; ++c)
		//{
		//	dataHigh[b][c][i][j] =
		//		m * dataIn[b][c][i][j] +
		//		(1 - m) * (N > 0 ? d[c] / N : 0);
		//}

		//ADJOINT

		scalar_t adjD[MAX_CHANNELS] = { 0 };
		scalar_t adjMask = 0;
		scalar_t adjN = 0;
		for (int c = 0; c < C; ++c)
		{
			const scalar_t adjDataHigh = gradDataHighIn[b][c][i][j];
			//dataHigh[b][c][i][j] = m * dataIn[b][c][i][j] + (1 - m) * (N > 0 ? d[c] / N : 0);
			adjMask += adjDataHigh * (dataIn[b][c][i][j] - (N > 0 ? d[c] / N : 0));
			gradDataOut[b][c][i][j] += adjDataHigh * m;
			adjN -= N > 0 ? adjDataHigh * (1 - m) * d[c] / (N * N) : 0;
			adjD[c] += N > 0 ? adjDataHigh * (1 - m) / N : 0;
		}
		const scalar_t adjMaskHigh = gradMaskHighIn[b][i][j];
		//maskHigh[b][i][j] = m + (1 - m) * (N / Weight);
		adjMask += adjMaskHigh * (1 - (N / Weight));
		adjN += adjMaskHigh * ((1 - m) / Weight);
		gradMaskOut[b][i][j] += adjMask;

#define ITEM(ii,jj,w)																					\
			if ((ii) >= 0 && (jj) >= 0 && (ii) < oH && (jj) < oW) {										\
				scalar_t adjMaskLow = 0;																\
				for (int c=0; c<C; ++c)																	\
				{																						\
					/* d[c] += w * maskLowIn[b][(ii)][(jj)] * dataLowIn[b][c][(ii)][(jj)]; */			\
					adjMaskLow += adjD[c] * w * dataLowIn[b][c][(ii)][(jj)];							\
					atomicAdd(gradDataLowOut.data() + IDX4(gradDataLowOut, b, c, (ii), (jj)),			\
						adjD[c] * w * maskLowIn[b][(ii)][(jj)]);										\
				}																						\
				/* N += w * maskLowIn[b][(ii)][(jj)]; */												\
				adjMaskLow += adjN * w;																	\
				atomicAdd(gradMaskLowOut.data() + IDX3(gradMaskLowOut, b, (ii), (jj)), adjMaskLow);		\
			}
		ITEM(i / 2 + io, j / 2 + jo, 0.25f * 0.25f);
		ITEM(i / 2, j / 2 + jo, 0.25f * 0.75f);
		ITEM(i / 2 + io, j / 2, 0.25f * 0.75f);
		ITEM(i / 2, j / 2, 0.75f * 0.75f);
#undef ITEM
	}
	KERNEL_3D_LOOP_END
}

#undef IDX3
#undef IDX4

template<typename scalar_t>
__global__ void Add3(dim3 virtual_size, Tensor3RW<scalar_t> dst, Tensor3Read<scalar_t> src)
{
	KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of high resolution
	{
		dst[b][i][j] += src[b][i][j];
	}
	KERNEL_3D_LOOP_END
}
template<typename scalar_t>
__global__ void Add4(dim3 virtual_size, Tensor4RW<scalar_t> dst, Tensor4Read<scalar_t> src)
{
	const int C = dst.size(1);
	KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of high resolution
	{
		for (int c=0; c<C; ++c)
		    dst[b][c][i][j] += src[b][c][i][j];
	}
	KERNEL_3D_LOOP_END
}

QUICKMLP_KERNEL_NAMESPACE_END
