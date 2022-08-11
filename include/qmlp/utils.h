#pragma once

#include "common.h"
#include "tensor.h"
#include "tensor.h"
#include "ckl/kernel_loader.h"

#include <stack>
#include <tuple>

QUICKMLP_NAMESPACE_BEGIN

/**
 * A collection of helper operations that do not fit in other categories.
 */
class Utils
{
    struct ForwardKernel
    {
        ckl::KernelFunction up;
        ckl::KernelFunction down;
    };
    struct AdjointKernel
    {
        ckl::KernelFunction up;
        ckl::KernelFunction down;
        ckl::KernelFunction add3;
        ckl::KernelFunction add4;
    };
    static std::optional<ForwardKernel> fractionalPullpushForwardKernel_[];
    static std::optional<AdjointKernel> fractionalPullpushAdjointKernel_[];


    static void compileFractionalPullpushKernel(Tensor::Precision p);
    static void fractionalPullpush_recursion(
        const Tensor& maskInput, const Tensor& dataInput,
        Tensor& maskOutput, Tensor& dataOutput,
        bool saveResults,
        std::stack<Tensor>& iStack,
        CUstream stream);

    static void compileFractionalPullpushAdjointKernel(Tensor::Precision p);
    static void adjointFractionalPullpush_recursion(
        const Tensor& maskInput, const Tensor& dataInput,
        const Tensor& gradMaskOutput, const Tensor& gradDataOutput,
        Tensor& gradMaskInput, Tensor& gradDataInput,
        std::stack<Tensor>& iStack,
        CUstream stream);

public:
    /**
     * \brief Applies fast Pullpush via down- and upsampling
	 * with fractional masks. This has a similar effect as the adaptive smoothing.
	 * All tensors must reside on the GPU and are of type float or double.
	 *
	 * The mask is defined as:
	 *  - 1: non-empty pixel
	 *  - 0: empty pixel
	 *  and any fraction in between.
	 *
     * \param mask the mask of shape (Batch, Height, Width)
     * \param data  the data of shape (Batch, Channels, Height, Width)
     * \param output the inpainted data of shape (Batch, Channels, Height, Width)
     */
    static void fractionalPullpush(
        const Tensor& mask, const Tensor& data,
        Tensor& output,
        CUstream stream);

    /**
     * \brief Adjoint code for \ref fractionalPullpush
     * \param mask the input mask of shape (Batch, Height, Width)
     * \param data the input data of shape (Batch, Channels, Height, Width)
     * \param adjOutput the gradient of the output of shape (Batch, Channels, Height, Width)
     * \param adjMask the gradient of the input mask of shape (Batch, Height, Width),
	 *		should be initialized with zero
     * \param adjData the gradient of the input data of shape (Batch, Channels, Height, Width),
	 *		should be initialized with zero
     */
    static void adjointFractionalPullpush(
        const Tensor& mask, const Tensor& data,
        const Tensor& adjOutput,
        Tensor& adjMask, Tensor& adjData,
        CUstream stream);
};

QUICKMLP_NAMESPACE_END
