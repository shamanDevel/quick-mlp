#pragma once

#include <cuda_fp16.h>
#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

namespace qmlp { namespace kernel {

template<int StartChannel, int NumChannels>
struct EncodingIdentity
{
    template<typename I>
    static __device__ void forward(const I input, half* output)
    {
#pragma unroll
        for (int i=0; i<NumChannels; ++i)
        {
            output[i] = __float2half(input[i + StartChannel]);
        }
    }

    template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O>
    static __device__ void adjoint(const I& input, const half* adjOutput, O& adjInput)
    {
        if constexpr(EvaluateInputGradients)
        {
#pragma unroll
            for (int i = 0; i < NumChannels; ++i)
            {
                adjInput[i + StartChannel] += __half2float(adjOutput[i]);
            }
        }
    }
};


}}
