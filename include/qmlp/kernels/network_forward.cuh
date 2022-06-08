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

__global__ void NetworkKernelInferenceAndForward(
    int numel,
    const Tensor2Read<float> inputs, //shape (numel, Cin)
    Tensor2RW<float> outputs, //shape (numel, Cout)
    const half* networkParameters,
    half* forwardTmpMemory //for storing the intermediate results in the forward pass
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
    const half hZERO = __float2half(0);

    KERNEL_1D_LOOP_SYNC(index, valid, numel)
    {
        //storage for the intermediate results
        half* intermediateResultsWarp = sIntermediateResults + 32 * MAX_CHANNELS * warpID;
        half* intermediateResultsThread = sIntermediateResults + MAX_CHANNELS * threadIdx.x;

        //encodings
        if (valid) {
            auto encodingInput = inputs[index];
            half* encodingOutput = intermediateResultsThread;
            //called e.g. EncodingIdentity::forward(encodingInput, encodingOutput)

            //CODE GENERATION [[
$$CALL_ENCODINGS$$
            //]] CODE GENERATIION

            //padding
            for (int cin = INPUT_PAD_START; cin < CHANNELS_IN; ++cin)
            {
                intermediateResultsThread[cin] = hZERO;
            }
        }
        else
        {
            //invalid index, fill with zeros to avoid NaNs
            for (int cin = 0; cin < CHANNELS_IN; ++cin)
            {
                intermediateResultsThread[cin] = hZERO;
            }
        }

        //call layers
        //e.g. qmlp::kernel::Layer<InChannelsDiv16, OutChannelsDiv16, MAX_CHANNELS, Bias, Activation>
        //          ::inference(networkParameters+offsetWeights, networkParameters+offsetBias, intermediateResultsWarp);
        //TODO: prefetch weights in shared memory?

        //CODE GENERATION [[
$$CALL_NETWORK_LAYERS$$
        //]] CODE GENERATIION

        //copy output
        if (valid)
        {
            for (int cout=0; cout<CHANNELS_OUT; ++cout)
            {
                outputs[index][cout] == __half2float(
                    intermediateResultsThread[cout]);
            }
        }
    }
    KERNEL_1D_LOOP_END
}

}}

