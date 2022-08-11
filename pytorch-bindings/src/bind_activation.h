#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include "../qmlp/bindings.h"
#include <qmlp/qmlp.h>
#include <qmlp/activation.h>


struct ActivationCache
{
    static QUICKMLP_NAMESPACE::Activation_ptr create(const std::string& cfg);
};

class ActivationBindings : public torch::CustomClassHolder
{
    std::string cfg_;
    QUICKMLP_NAMESPACE::Activation_ptr a_;

public:
    ActivationBindings(const std::string& cfg)
        : cfg_(cfg), a_(ActivationCache::create(cfg))
    {}

    [[nodiscard]] std::string cfg() const
    {
        return cfg_;
    }

    //no gradients
    torch::Tensor inference(const torch::Tensor& input) const;

    torch::Tensor adjoint(const torch::Tensor& input, const torch::Tensor& adjOutput) const;

    //with autograd support
    static torch::Tensor forward(c10::intrusive_ptr<ActivationBindings> self, const torch::Tensor& input);
};
typedef c10::intrusive_ptr<ActivationBindings> ActivationBindings_ptr;

struct ActivationAutogradFunction : public torch::autograd::Function<ActivationAutogradFunction>
{
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        ActivationBindings_ptr activ);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs);
};

void bindActivation(torch::Library& m);
