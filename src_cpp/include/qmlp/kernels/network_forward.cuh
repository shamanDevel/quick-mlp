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



__global__ void NetworkKernelInferenceAndForward(
    int numel,
    const Tensor2Read<float> inputs, //shape (numel, Cin)
    Tensor2RW<float> outputs, //shape (numel, Cout)
    const half* networkParameters,
    half* forwardTmpMemory //for storing the intermediate results in the forward pass
    )
{
    //The maximal channels, including the skew --> the stride between items
    constexpr int MAX_CHANNELS = $$MAX_CHANNELS$;

    //shared memory for the intermediate states
    extern __shared__ half sIntermediateResults[];

    const int warpID = threadIdx.x / 32;
    const int lineID = threadIdx.x % 32;
    const int numWarps = blockDim.x / 32;
    const int numel32 = roundUpPower2(numel, 32);

    constexpr int INPUT_PAD_START = $$INPUT_PAD_START$$;
    constexpr int CHANNELS_IN = $$CHANNELS_IN$$; // these are multiples of the warp size (16)
    constexpr int CHANNELS_OUT = $$CHANNELS_OUT$$;

    KERNEL_1D_LOOP_SYNC(index, valid, numel)
    {
        //global warp index, used to index the forward tmp memory
        const int globalWarpID = index / 32;

        //storage for the intermediate results
        half* intermediateResultsWarp = sIntermediateResults + 32 * MAX_CHANNELS * warpID;
        half* intermediateResultsThread = sIntermediateResults + MAX_CHANNELS * threadIdx.x;

        //encodings
        if (valid) {
            auto encodingInput = inputs[index];
            WrappedArray<half> encodingOutput{ intermediateResultsThread, MAX_CHANNELS};
            //called e.g. EncodingIdentity::forward(encodingInput, encodingOutput)

            //CODE GENERATION [[
$$CALL_ENCODINGS$$
            //]] CODE GENERATIION

            //padding
            for (int cin = INPUT_PAD_START; cin < CHANNELS_IN; ++cin)
            {
                intermediateResultsThread[cin] = hZERO();
            }
        }
        else
        {
            //invalid index, fill with zeros to avoid NaNs
            for (int cin = 0; cin < CHANNELS_IN; ++cin)
            {
                intermediateResultsThread[cin] = hZERO();
            }
        }
        __syncwarp();

        //call layers
        //e.g. qmlp::kernel::Layer<InChannelsDiv16, OutChannelsDiv16, MAX_CHANNELS, Bias, Activation>
        //          ::inference(networkParameters+offsetWeights, networkParameters+offsetBias, intermediateResultsWarp);
        //TODO: prefetch weights in shared memory?

        //CODE GENERATION [[
$$CALL_NETWORK_LAYERS$$
        //]] CODE GENERATIION

        //copy output
        __syncwarp();
        if (valid)
        {
            const int actual_channels_out = outputs.size(1);
            for (int cout=0; cout<actual_channels_out; ++cout)
            {
                outputs[index][cout] = __half2float(
                    intermediateResultsThread[cout]);
            }
        }
    }
    KERNEL_1D_LOOP_END
}

QUICKMLP_KERNEL_NAMESPACE_END
