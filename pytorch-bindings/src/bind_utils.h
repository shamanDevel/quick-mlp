#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include <qmlp/utils.h>
#include <qmlp/qmlp.h>
#include "../qmlp/bindings.h"


class UtilsBindings : public torch::CustomClassHolder
{

public:

    //forward, no gradients
    [[nodiscard]] static torch::Tensor pullpushForward(
        const torch::Tensor& mask, const torch::Tensor& data);

    //adjoint
    [[nodiscard]] static std::tuple<torch::Tensor, torch::Tensor> pullpushAdjoint(
        const torch::Tensor& mask, const torch::Tensor& data,
        const torch::Tensor& adjOutput);

    //forward with gradients
    [[nodiscard]] static torch::Tensor pullpush(
        const torch::Tensor& mask, const torch::Tensor& data);
    
};

struct PushPullAutogradFunction : public torch::autograd::Function<PushPullAutogradFunction>
{
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& mask, const torch::Tensor& data);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs);
};


void bindUtils(torch::Library& m);
