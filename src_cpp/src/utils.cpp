#include <tinyformat.h>
#include <qmlp/utils.h>

#include "qmlp/qmlp.h"
#include "qmlp/kernels/tensor.cuh"

QUICKMLP_NAMESPACE_BEGIN

std::optional<Utils::ForwardKernel> Utils::fractionalPullpushForwardKernel_[Tensor::Precision::_NUM_PRECISION_];
std::optional<Utils::AdjointKernel> Utils::fractionalPullpushAdjointKernel_[Tensor::Precision::_NUM_PRECISION_];

void Utils::compileFractionalPullpushKernel(Tensor::Precision p)
{
    if (!fractionalPullpushForwardKernel_[p].has_value())
    {
        auto kl = QuickMLP::Instance().kernelLoader();
        auto file = ckl::KernelLoader::MainFile("qmlp/kernels/pullpush.cuh");
        int compileFlags = QuickMLP::Instance().getCompileFlags();

        fractionalPullpushForwardKernel_[p] = {
            kl->getKernel(
                tinyformat::format(
                    "qmlp::kernel::FastInpaintingFractionalKernel_Up<%s>",
                    Tensor::DatatypePerEntry[p]
                ),
                file, {}, compileFlags).value(),
            kl->getKernel(
                tinyformat::format(
                    "qmlp::kernel::FastInpaintingFractionalKernel_Down<%s>",
                    Tensor::DatatypePerEntry[p]
                ),
                file, {}, compileFlags).value()
        };
    }
}

void Utils::fractionalPullpush_recursion(const Tensor& maskInput, const Tensor& dataInput, Tensor& maskOutput,
    Tensor& dataOutput, bool saveResults, std::stack<Tensor>& iStack, CUstream stream)
{
    const auto B = dataInput.size(0);
    const auto C = dataInput.size(1);
    const auto H = dataInput.size(2);
    const auto W = dataInput.size(3);
    const auto dtype = dataInput.precision();

    if (H <= 1 || W <= 1)
    {
        //end of recursion
        maskOutput.copyAsync_(maskInput, stream);
        dataOutput.copyAsync_(dataInput, stream);
        return;
    }

    const auto oH = H / 2;
    const auto oW = W / 2;

    //downsample
    Tensor maskLowPre(dtype, { B, oH, oW });
    Tensor dataLowPre(dtype, { B, C, oH, oW });
    QUICKMLP_DISPATCH_FLOATING_TYPES(dtype, "FastInpaintingFractionalKernel_Down", ([&]
        {
            auto fun = fractionalPullpushForwardKernel_[dtype].value().down;
            dim3 virtual_size(oW, oH, B);
            int numel = oW * oH * B;
            int minGridSize = std::min(
                CKL_DIV_UP(numel, fun.bestBlockSize()),
                fun.minGridSize());
            auto maskAcc = maskInput.accessor<kernel::Tensor3Read<scalar_t>>();
            auto dataAcc = dataInput.accessor<kernel::Tensor4Read<scalar_t>>();
            auto maskLowAcc = maskLowPre.accessor<kernel::Tensor3RW<scalar_t>>();
            auto dataLowAcc = dataLowPre.accessor<kernel::Tensor4RW<scalar_t>>();
            fun.call(minGridSize, fun.bestBlockSize(), 0, stream,
                virtual_size, maskAcc, dataAcc, maskLowAcc, dataLowAcc);
        }));

    //recursion
    Tensor maskLowPost(dtype, { B, oH, oW });
    Tensor dataLowPost(dtype, { B, C, oH, oW });
    fractionalPullpush_recursion(maskLowPre, dataLowPre, maskLowPost, dataLowPost, saveResults, iStack, stream);

    //upsample
    QUICKMLP_DISPATCH_FLOATING_TYPES(dtype, "FastInpaintingKernel_Up", ([&]
        {
            auto fun = fractionalPullpushForwardKernel_[dtype].value().up;
            dim3 virtual_size(W, H, B);
            int numel = W * H * B;
            int minGridSize = std::min(
                CKL_DIV_UP(numel, fun.bestBlockSize()),
                fun.minGridSize());

            auto maskAcc = maskInput.accessor<kernel::Tensor3Read<scalar_t>>();
            auto dataAcc = dataInput.accessor<kernel::Tensor4Read<scalar_t>>();
            auto maskLowAcc = maskLowPost.accessor<kernel::Tensor3Read<scalar_t>>();
            auto dataLowAcc = dataLowPost.accessor<kernel::Tensor4Read<scalar_t>>();
            auto maskHighAcc = maskOutput.accessor<kernel::Tensor3RW<scalar_t>>();
            auto dataHighAcc = dataOutput.accessor<kernel::Tensor4RW<scalar_t>>();

            fun.call(minGridSize, fun.bestBlockSize(), 0, stream,
                virtual_size, maskAcc, dataAcc, maskLowAcc, dataLowAcc, maskHighAcc, dataHighAcc);
        }));

    //save for adjoint
    if (saveResults)
    {
        iStack.push(maskLowPre);
        iStack.push(dataLowPre);
        iStack.push(maskLowPost);
        iStack.push(dataLowPost);
    }
}

void Utils::fractionalPullpush(const Tensor& mask, const Tensor& data, Tensor& output, CUstream stream)
{
    CHECK_DIM(mask, 3);
    CHECK_DIM(data, 4);
    CHECK_DIM(output, 4);

    auto B = data.size(0);
    auto C = data.size(1);
    auto H = data.size(2);
    auto W = data.size(3);

    CHECK_SIZE(mask, 0, B);
    CHECK_SIZE(mask, 1, H);
    CHECK_SIZE(mask, 2, W);
    CHECK_SIZE(output, 0, B);
    CHECK_SIZE(output, 1, C);
    CHECK_SIZE(output, 2, H);
    CHECK_SIZE(output, 3, W);

    auto dtype = data.precision();
    CHECK_DTYPE(mask, dtype);
    CHECK_DTYPE(output, dtype);

    CHECK_ERROR(C < 16, "fractionalPullpush only supports up to 16 channels, but got " + std::to_string(C));

    compileFractionalPullpushKernel(dtype);

    //inpaint recursivly
    std::stack<Tensor> s;
    Tensor maskOutput(dtype, mask.sizes());
    fractionalPullpush_recursion(mask, data, maskOutput, output, false, s, stream);
}

void Utils::compileFractionalPullpushAdjointKernel(Tensor::Precision p)
{
    if (!fractionalPullpushAdjointKernel_[p].has_value())
    {
        auto kl = QuickMLP::Instance().kernelLoader();
        auto file = ckl::KernelLoader::MainFile("qmlp/kernels/pullpush.cuh");
        int compileFlags = QuickMLP::Instance().getCompileFlags();

        fractionalPullpushAdjointKernel_[p] = {
            kl->getKernel(
                tinyformat::format(
                    "qmlp::kernel::AdjFastInpaintingFractionalKernel_Up<%s>",
                    Tensor::DatatypePerEntry[p]
                ),
                file, {}, compileFlags).value(),
            kl->getKernel(
                tinyformat::format(
                    "qmlp::kernel::AdjFastInpaintingFractionKernel_Down<%s>",
                    Tensor::DatatypePerEntry[p]
                ),
                file, {}, compileFlags).value(),
            kl->getKernel(
                tinyformat::format(
                    "qmlp::kernel::Add3<%s>",
                    Tensor::DatatypePerEntry[p]
                ),
                file, {}, compileFlags).value(),
            kl->getKernel(
                tinyformat::format(
                    "qmlp::kernel::Add4<%s>",
                    Tensor::DatatypePerEntry[p]
                ),
                file, {}, compileFlags).value()
        };
    }
}

void Utils::adjointFractionalPullpush_recursion(const Tensor& maskInput, const Tensor& dataInput,
    const Tensor& gradMaskOutput, const Tensor& gradDataOutput, Tensor& gradMaskInput, Tensor& gradDataInput,
    std::stack<Tensor>& iStack, CUstream stream)
{
    const auto B = dataInput.size(0);
    const auto C = dataInput.size(1);
    const auto H = dataInput.size(2);
    const auto W = dataInput.size(3);
    const auto oH = H / 2;
    const auto oW = W / 2;
    const auto dtype = dataInput.precision();

    if (H <= 1 || W <= 1)
    {
        //end of recursion
        QUICKMLP_DISPATCH_FLOATING_TYPES(dtype, "AdjFastInpaintingFractionKernel_Add", ([&]
            {
                dim3 virtual_size(W, H, B);
                int numel = W * H * B;

                {
                    auto fun = fractionalPullpushAdjointKernel_[dtype].value().add3;
                    int minGridSize = std::min(
                        CKL_DIV_UP(numel, fun.bestBlockSize()),
                        fun.minGridSize());
                    auto accDst = gradMaskInput.accessor<kernel::Tensor3RW<scalar_t>>();
                    auto accSrc = gradMaskOutput.accessor<kernel::Tensor3Read<scalar_t>>();
                    fun.call(minGridSize, fun.bestBlockSize(), 0, stream,
                        virtual_size, accDst, accSrc);
                }

                {
                    auto fun = fractionalPullpushAdjointKernel_[dtype].value().add4;
                    int minGridSize = std::min(
                        CKL_DIV_UP(numel, fun.bestBlockSize()),
                        fun.minGridSize());
                    auto accDst = gradDataInput.accessor<kernel::Tensor4RW<scalar_t>>();
                    auto accSrc = gradDataOutput.accessor<kernel::Tensor4Read<scalar_t>>();
                    fun.call(minGridSize, fun.bestBlockSize(), 0, stream,
                        virtual_size, accDst, accSrc);
                }
            }));
        
        return;
    }

    //get saved tensors (from after recursion in the forward pass
    Tensor dataLowPost = iStack.top(); iStack.pop();
    Tensor maskLowPost = iStack.top(); iStack.pop();
    Tensor dataLowPre = iStack.top(); iStack.pop();
    Tensor maskLowPre = iStack.top(); iStack.pop();

    //adjoint upsample
    Tensor gradMaskLowPost(dtype, { B, oH, oW });
    Tensor gradDataLowPost(dtype, { B, C, oH, oW });
    gradMaskLowPost.zero_();
    gradDataLowPost.zero_();
    QUICKMLP_DISPATCH_FLOATING_TYPES(dtype, "AdjFastInpaintingFractionalKernel_Up", ([&]
        {
            auto fun = fractionalPullpushAdjointKernel_[dtype].value().up;
            dim3 virtual_size(W, H, B);
            int numel = W * H * B;
            int minGridSize = std::min(
                CKL_DIV_UP(numel, fun.bestBlockSize()),
                fun.minGridSize());

            auto maskInAcc = maskInput.accessor<kernel::Tensor3Read<scalar_t>>();
            auto dataInAcc = dataInput.accessor<kernel::Tensor4Read<scalar_t>>();
            auto maskLowInAcc = maskLowPost.accessor<kernel::Tensor3Read<scalar_t>>();
            auto dataLowInAcc = dataLowPost.accessor<kernel::Tensor4Read<scalar_t>>();
            auto gradMaskHighInAcc = gradMaskOutput.accessor<kernel::Tensor3Read<scalar_t>>();
            auto gradDataHighInAcc = gradDataOutput.accessor<kernel::Tensor4Read<scalar_t>>();

            auto gradMaskOutAcc = gradMaskInput.accessor<kernel::Tensor3RW<scalar_t>>();
            auto gradDataOutAcc = gradDataInput.accessor<kernel::Tensor4RW<scalar_t>>();
            auto gradMaskLowOutAcc = gradMaskLowPost.accessor<kernel::Tensor3RW<scalar_t>>();
            auto gradDataLowOutAcc = gradDataLowPost.accessor<kernel::Tensor4RW<scalar_t>>();

            fun.call(minGridSize, fun.bestBlockSize(), 0, stream,
                virtual_size,
                maskInAcc, dataInAcc, maskLowInAcc, dataLowInAcc, gradMaskHighInAcc, gradDataHighInAcc,
                gradMaskOutAcc, gradDataOutAcc, gradMaskLowOutAcc, gradDataLowOutAcc);
        }));

    //recursion
    Tensor gradMaskLowPre(dtype, { B, H, W });
    Tensor gradDataLowPre(dtype, { B, C, H, W });
    gradMaskLowPre.zero_();
    gradDataLowPre.zero_();
    adjointFractionalPullpush_recursion(
        maskLowPre,
        dataLowPre,
        gradMaskLowPost,
        gradDataLowPost,
        gradMaskLowPre,
        gradDataLowPre,
        iStack, stream);

    //adjoint downsample
    QUICKMLP_DISPATCH_FLOATING_TYPES(dtype, "AdjFastInpaintingFractionKernel_Down", ([&]
        {
            auto fun = fractionalPullpushAdjointKernel_[dtype].value().down;
            dim3 virtual_size(oW, oH, B);
            int numel = oW * oH * B;
            int minGridSize = std::min(
                CKL_DIV_UP(numel, fun.bestBlockSize()),
                fun.minGridSize());

            auto maskInAcc = maskInput.accessor<kernel::Tensor3Read<scalar_t>>();
            auto dataInAcc = dataInput.accessor<kernel::Tensor4Read<scalar_t>>();
            auto gradMaskLowPreAcc = gradMaskLowPre.accessor<kernel::Tensor3Read<scalar_t>>();
            auto gradDataLowPreAcc = gradDataLowPre.accessor<kernel::Tensor4Read<scalar_t>>();

            auto gradMaskOutAcc = gradMaskInput.accessor<kernel::Tensor3RW<scalar_t>>();
            auto gradDataOutAcc = gradDataInput.accessor<kernel::Tensor4RW<scalar_t>>();

            fun.call(minGridSize, fun.bestBlockSize(), 0, stream,
                virtual_size,
                maskInAcc, dataInAcc, gradMaskLowPreAcc, gradDataLowPreAcc,
                gradMaskOutAcc, gradDataOutAcc);
        }));
}

void Utils::adjointFractionalPullpush(const Tensor& mask, const Tensor& data, const Tensor& adjOutput,
                                      Tensor& adjMask, Tensor& adjData, CUstream stream)
{
    CHECK_DIM(mask, 3);
    CHECK_DIM(data, 4);
    CHECK_DIM(adjOutput, 4);
    CHECK_DIM(adjMask, 3);
    CHECK_DIM(adjData, 4);

    auto B = data.size(0);
    auto C = data.size(1);
    auto H = data.size(2);
    auto W = data.size(3);

    CHECK_SIZE(mask, 0, B);
    CHECK_SIZE(mask, 1, H);
    CHECK_SIZE(mask, 2, W);
    CHECK_SIZE(adjOutput, 0, B);
    CHECK_SIZE(adjOutput, 1, C);
    CHECK_SIZE(adjOutput, 2, H);
    CHECK_SIZE(adjOutput, 3, W);
    CHECK_SIZE(adjMask, 0, B);
    CHECK_SIZE(adjMask, 1, H);
    CHECK_SIZE(adjMask, 2, W);
    CHECK_SIZE(adjData, 0, B);
    CHECK_SIZE(adjData, 1, C);
    CHECK_SIZE(adjData, 2, H);
    CHECK_SIZE(adjData, 3, W);

    auto dtype = data.precision();
    CHECK_DTYPE(mask, dtype);
    CHECK_DTYPE(adjOutput, dtype);
    CHECK_DTYPE(adjMask, dtype);
    CHECK_DTYPE(adjData, dtype);

    CHECK_ERROR(C < 16, "fractionalPullpush only supports up to 16 channels, but got " + std::to_string(C));

    compileFractionalPullpushKernel(dtype);
    compileFractionalPullpushAdjointKernel(dtype);

    //run forward again, but save results
    std::stack<Tensor> s;
    Tensor maskOutput(dtype, mask.sizes());
    Tensor output(dtype, adjOutput.sizes());
    fractionalPullpush_recursion(mask, data, maskOutput, output, true, s, stream);

    //run adjoint code
    Tensor gradMaskOutput(dtype, mask.sizes());
    gradMaskOutput.zero_();
    adjointFractionalPullpush_recursion(mask, data, gradMaskOutput, adjOutput, adjMask, adjData, s, stream);
    
}

QUICKMLP_NAMESPACE_END
