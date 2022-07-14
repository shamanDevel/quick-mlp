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

namespace activations
{
    $$DEFINE_ACTIVATIONS$$
}

__global__ void ActivationForwardKernel(
    dim3 virtual_size, //numel, C, 1
    const Tensor2Read<half> inputs, //shape (numel, C)
    Tensor2RW<half> outputs //shape (numel, C)
    )
{
    typedef $$ACTIVATION_ID$$ Activation_t;

    KERNEL_2D_LOOP(i, j, virtual_size)
    {
        half in = inputs[i][j];
        half out = Activation_t::forward(in);
        outputs[i][j] = out;
    }
    KERNEL_2D_LOOP_END
}

__global__ void ActivationAdjointKernel(
    dim3 virtual_size, //numel, C, 1
    const Tensor2Read<half> inputs, //shape (numel, C)
    const Tensor2Read<half> adjOutputs, //shape (numel, C)
    Tensor2RW<half> adjInputs //shape (numel, C)
)
{
    typedef $$ACTIVATION_ID$$ Activation_t;

    KERNEL_2D_LOOP(i, j, virtual_size)
    {
        half in = inputs[i][j];
        half adjOut = adjOutputs[i][j];
        half adjIn = Activation_t::adjoint(in, adjOut);
        adjInputs[i][j] = adjIn;
    }
    KERNEL_2D_LOOP_END
}

QUICKMLP_KERNEL_NAMESPACE_END
