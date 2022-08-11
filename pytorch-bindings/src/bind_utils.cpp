#include "bind_utils.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>


torch::Tensor UtilsBindings::pullpushForward(const torch::Tensor& mask, const torch::Tensor& data)
{
    auto output = torch::empty_like(data);

    auto maskWrapped = QUICKMLP_NAMESPACE::wrap(mask);
    auto dataWrapped = QUICKMLP_NAMESPACE::wrap(data);
    auto outputWrapped = QUICKMLP_NAMESPACE::wrap(output);

    CUstream stream = c10::cuda::getCurrentCUDAStream();
    QUICKMLP_NAMESPACE::Utils::fractionalPullpush(maskWrapped, dataWrapped, outputWrapped, stream);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> UtilsBindings::pullpushAdjoint(
    const torch::Tensor& mask,
    const torch::Tensor& data, const torch::Tensor& adjOutput)
{
    auto gradMask = torch::zeros_like(mask);
    auto gradData = torch::zeros_like(data);

    auto maskWrapped = QUICKMLP_NAMESPACE::wrap(mask);
    auto dataWrapped = QUICKMLP_NAMESPACE::wrap(data);
    auto adjOutputWrapped = QUICKMLP_NAMESPACE::wrap(adjOutput);
    auto gradMaskWrapped = QUICKMLP_NAMESPACE::wrap(gradMask);
    auto gradDataWrapped = QUICKMLP_NAMESPACE::wrap(gradData);

    CUstream stream = c10::cuda::getCurrentCUDAStream();
    QUICKMLP_NAMESPACE::Utils::adjointFractionalPullpush(
        maskWrapped, dataWrapped, adjOutputWrapped, gradMaskWrapped, gradDataWrapped, stream);

    return std::make_tuple(gradMask, gradData);
}

torch::Tensor UtilsBindings::pullpush(const torch::Tensor& mask, const torch::Tensor& data)
{
    auto ret = PushPullAutogradFunction::apply(mask, data);
    return ret[0];
}

torch::autograd::variable_list PushPullAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
    const torch::Tensor& mask, const torch::Tensor& data)
{
    ctx->save_for_backward({ mask, data });
    torch::Tensor output = UtilsBindings::pullpushForward(mask, data);
    return { output };
}

torch::autograd::tensor_list PushPullAutogradFunction::backward(torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    torch::Tensor mask = saved[0];
    torch::Tensor data = saved[1];
    torch::Tensor adjOutput = grad_outputs[0];

    auto ret = UtilsBindings::pullpushAdjoint(mask, data, adjOutput);
    return { std::get<0>(ret), std::get<1>(ret) };
}

void bindUtils(torch::Library& m)
{
    m.class_<UtilsBindings>("utils")
        .def_static("pullpush_forward", &UtilsBindings::pullpushForward)
        .def_static("pullpush_backward", &UtilsBindings::pullpushAdjoint)
        .def_static("pullpush", &UtilsBindings::pullpush, R"doc(
Performs differentiable inpainting via the Pull-Push algorithm.

All tensors must reside on the GPU and are of type float or double.
The mask is defined as:
 - 1: non-empty pixel
 - 0: empty pixel
 and any fraction in between.

:param mask: the mask of shape (Batch, Height, Width)
:param data: the data of shape (Batch, Channels, Height, Width)
:output: the inpainted data of shape (Batch, Channels, Height, Width)
)doc")
        ;
}
