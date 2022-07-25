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
#include <qmlp/kernels/layer.cuh>
#include <qmlp/kernels/tensor.cuh>
#include <qmlp/kernels/loops.cuh>

$$INCLUDES$$

$$ENCODING_CONSTANTS$$

QUICKMLP_KERNEL_NAMESPACE_BEGIN

namespace activations
{
    $$DEFINE_ACTIVATIONS$$
}

//template<bool HasInputGradients, bool HasNetworkGradients, bool HasEncodingGradients>
__global__ void NetworkKernelBackward(
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
    const int numel32 = roundUpPower2(numel, 32);

    constexpr int INPUT_PAD_START = $$INPUT_PAD_START$$;
    constexpr int CHANNELS_IN = $$CHANNELS_IN$$;
    constexpr int CHANNELS_OUT = $$CHANNELS_OUT$$;
    constexpr bool HAS_INPUT_GRADIENTS = $$HAS_INPUT_GRADIENTS$$;

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
        else
        {
            for (int cout = 0; cout < CHANNELS_OUT; ++cout)
                adjIntermediateResultsThread[cout] = hZERO();
        }
        __syncwarp();
        //printf("index=%d, valid=%d\n", int(index), valid?1:0);

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
        __syncwarp();
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

QUICKMLP_KERNEL_NAMESPACE_END
