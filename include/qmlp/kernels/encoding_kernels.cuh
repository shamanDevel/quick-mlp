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

$$INCLUDES$$
$$ENCODING_CONSTANTS$$

QUICKMLP_KERNEL_NAMESPACE_BEGIN


__global__ void EncodingForwardKernel(
    long long virtual_size, //numel, 1, 1
    const Tensor2Read<float> inputs, //shape (numel, C)
    Tensor2RW<half> outputs //shape (numel, C)
    )
{
    KERNEL_1D_LOOP(index, virtual_size)
    {
        auto encodingInput = inputs[index];
        auto encodingOutput = outputs[index].data(); //assume stride of C = 1

        //CODE GENERATION [[
        $$CALL_ENCODINGS$$
        //]] CODE GENERATIION
    }
    KERNEL_1D_LOOP_END
}

__global__ void EncodingAdjointKernel(
    long long virtual_size, //numel, C, 1
    const Tensor2Read<float> inputs, //shape (numel, C)
    const Tensor2Read<half> adjOutputs, //shape (numel, C)
    Tensor2RW<float> adjInputs //shape (numel, C)
)
{
    KERNEL_1D_LOOP(index, virtual_size)
    {
        auto encodingInput = inputs[index];
        auto encodingAdjOutput = adjOutputs[index].data(); //assume stride of C = 1
        auto encodingAdjInput = adjInputs[index];

        //CODE GENERATION [[
        $$CALL_ENCODINGS$$
        //]] CODE GENERATIION
    }
}

QUICKMLP_KERNEL_NAMESPACE_END
