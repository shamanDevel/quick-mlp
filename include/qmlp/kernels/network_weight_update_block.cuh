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

#include <qmlp/kernels/layer.cuh>
#include <qmlp/kernels/tensor.cuh>
#include <qmlp/kernels/loops.cuh>

$$INCLUDES$$

$$ENCODING_CONSTANTS$$


namespace qmlp { namespace kernel {

namespace activations
{
    $$DEFINE_ACTIVATIONS$$
}

template<typename T, int N>
class StaticArray
{
    T data_[N];

public:
    __device__ StaticArray()
        : data_{0}
    {}

};

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

/**
 * \brief Loader for loading hat{O}_tmp into shared memory and registers.
 * This is used as the A-matrix for the matmul.
 */
template<int MDiv16>
struct OHatTmpLoader
{
    
};

/**
 * \brief Computes C += A*B^T block-synchronous.
 * The matrix A has shape MxK, matrix B has shape NxK.
 * M and N must be multiples of 16 and compile-time constants, specified by MDiv16, NDiv16.
 * K is flexible and can be large.
 *
 * 
 *
 * \tparam AccuT the type of C, can be half or float
 * \tparam ALoader the 
 * \tparam BLoader 
 * \tparam MDiv16 
 * \tparam NDiv16 
 * \param numel 
 * \param aLoader 
 * \param bLoader 
 * \param out 
 */
template<typename AccuT, typename ALoader, typename BLoader, int MDiv16, int NDiv16>
__device__ void matmulAxBt_block(int numel, const ALoader& aLoader, const BLoader& bLoader, AccuT* out)
{
    using namespace nvcuda::wmma;

    //per-warp: store the result of matrix C, shape MxN
    fragment<accumulator, 16, 16, 16, AccuT> c_frag[MDiv16][NDiv16];
    //bias
    StaticArray<AccuT, MDiv16> bias_frag;

    //matrices A and B


    //now loop over the partitions of K of size 32
    const int warpID = threadIdx.x / 32;
    const int lineID = threadIdx.x % 32;
    const int numWarps = blockDim.x / 32;
    KERNEL_1D_LOOP_BLOCKSYNC(index, valid, numel)
    {
        
    }
    KERNEL_1D_LOOP_END
}

__global__ void NetworkKernelWeightUpdateBlock(
    int numel,
    const Tensor2Read<float> inputs, //shape (numel, Cin)
    const Tensor2Read<float> adjOutputs, //shape (numel, Cout)
    Tensor2RW<float> adjInputs,
    const half* networkParameters,
    const half* forwardTmpMemory, //O_tmp from the forward pass
    half* adjointTmpMemory //hat(O_tmp), if weight gradients are requested else nullptr
    )
{
    constexpr int MAX_CHANNELS = $$MAX_CHANNELS$;
    //shared memory for the intermediate states
    extern __shared__ half sIntermediateResults[];

    const int warpID = threadIdx.x / 32;
    const int lineID = threadIdx.x % 32;
    const int numWarps = blockDim.x / 32;

    constexpr int INPUT_PAD_START = $$INPUT_PAD_START$$;
    constexpr int CHANNELS_IN = $$CHANNELS_IN$$;
    constexpr int CHANNELS_OUT = $$CHANNELS_OUT$$;
    constexpr bool HAS_INPUT_GRADIENTS = $$HAS_INPUT_GRADIENTS$$;
    const half hZERO = __float2half(0);

    KERNEL_1D_LOOP_SYNC(index, valid, numel)
    {
        //storage for the intermediate results
        half* adjIntermediateResultsWarp = sIntermediateResults + 32 * MAX_CHANNELS * warpID;
        half* adjIntermediateResultsThread = sIntermediateResults + MAX_CHANNELS * threadIdx.x;

        //fetch adj-output
        if (valid)
        {
            for (int cout = 0; cout < CHANNELS_OUT; ++cout)
            {
                adjIntermediateResultsThread[cout] = __float2half(
                    adjOutputs[index][cout]);
            }
        }

        //call layers
        //e.g. qmlp::kernel::Layer<InChannelsDiv16, OutChannelsDiv16, MAX_CHANNELS, Bias, Activation>
        //          ::template adjoint<ComputeWeightGradients>(
        //              networkParameters+offsetWeights, networkParameters+offsetBias, intermediateResultsWarp,
        //              forwardTmpMemory+offsetIntermediate, adjointTmpMemory+offsetIntermediate);
        //TODO: prefetch weights in shared memory?

        //CODE GENERATION [[
$$CALL_NETWORK_LAYERS$$
        //]] CODE GENERATIION

        //adjoint encodings
        if (valid) {
            auto encodingInput = inputs[index];
            TensorAccessor<float, 1, DefaultPtrTraits, int> adjEncodingInput;
            if constexpr (HAS_INPUT_GRADIENTS) {
                adjEncodingInput = adjInputs[index];
            }

            half* adjEncodingOutput = adjIntermediateResultsThread;

            //called e.g. EncodingIdentity::template adjoint<HAS_INPUT_GRADIENTS, parameterGradients>(
            //                     encodingInput, adjEncodingOutput, adjEncodingInput);
            //CODE GENERATION [[
$$CALL_ENCODINGS$$
            //]] CODE GENERATIION
        }
        
    }
    KERNEL_1D_LOOP_END
}

}}

