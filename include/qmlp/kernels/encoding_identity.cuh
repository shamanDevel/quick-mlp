#pragma once

#include <cuda_fp16.h>
#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

#include "common.cuh"

QUICKMLP_KERNEL_NAMESPACE_BEGIN

template<int StartChannel, int NumChannels>
struct EncodingIdentity
{
    template<typename I, typename O>
    static __device__ void forward(const I input, O* output)
    {
#pragma unroll
        for (int i=0; i<NumChannels; ++i)
        {
            output[i] = fcast<O>(input[i + StartChannel]);
        }
    }

    template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O, typename AdjI>
    static __device__ void adjoint(const I& input, const O& adjOutput, AdjI& adjInput)
    {
        if constexpr(EvaluateInputGradients)
        {
#pragma unroll
            for (int i = 0; i < NumChannels; ++i)
            {
                adjInput[i + StartChannel] += fcast<float>(adjOutput[i]);
            }
        }
    }
};

QUICKMLP_KERNEL_NAMESPACE_END
