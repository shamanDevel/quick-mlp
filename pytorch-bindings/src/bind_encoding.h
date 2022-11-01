#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include <qmlp/iencoding.h>
#include <qmlp/qmlp.h>
#include "../qmlp/bindings.h"


class EncodingBindings : public torch::CustomClassHolder
{
    std::string cfg_;
    QUICKMLP_NAMESPACE::IEncoding_ptr a_;

public:
    EncodingBindings(const std::string& cfg)
        : cfg_(cfg), a_()
    {
        nlohmann::json j = nlohmann::json::parse(cfg);
        a_ = QUICKMLP_NAMESPACE::EncodingFactory::Instance().create(j);
    }
    EncodingBindings(QUICKMLP_NAMESPACE::IEncoding_ptr a)
        : cfg_(), a_(a)
    {}

    [[nodiscard]] std::string cfg() const
    {
        return cfg_;
    }

    [[nodiscard]] int64_t maxInputChannel() const { return a_->maxInputChannel(); }
    [[nodiscard]] int64_t numOutputChannels() const { return a_->numOutputChannels(); }
    [[nodiscard]] int64_t hasParameters() const { return a_->hasParameters() ? 1 : 0; }
    [[nodiscard]] std::string toJson() const;
    /**
     * Creates an empty parameter fit for storing the inference parameters.
     * Note that the memory is uninitialized!
     */
    [[nodiscard]] torch::Tensor createInferenceParameters() const
    {
        TORCH_CHECK(a_->hasParameters());
        return torch::empty({ static_cast<long long>(a_->parameterCount()) },
            at::TensorOptions().dtype(
            a_->parameterPrecision(qmlp::Tensor::INFERENCE) == qmlp::Tensor::FLOAT ? c10::kFloat : c10::kHalf
        ).device(c10::kCUDA).requires_grad(true));
    }
    /**
     * Creates an empty parameter fit for storing the gradient parameters.
     * Note that the memory is uninitialized!
     */
    [[nodiscard]] torch::Tensor createGradientParameters() const
    {
        TORCH_CHECK(a_->hasParameters());
        return torch::empty({ static_cast<long long>(a_->parameterCount()) },
            at::TensorOptions().dtype(
                a_->parameterPrecision(qmlp::Tensor::GRADIENTS) == qmlp::Tensor::FLOAT ? c10::kFloat : c10::kHalf
            ).device(c10::kCUDA));
    }

    //no gradients
    torch::Tensor inference(const torch::Tensor& input) const;
    torch::Tensor inferenceWithParameter(const torch::Tensor& input, const torch::Tensor& parameterForward) const;

    torch::Tensor adjoint(const torch::Tensor& input, const torch::Tensor& adjOutput) const;
    //adjInput, adjParameterForward
    std::tuple<torch::Tensor, torch::Tensor> adjointWithParameter(
        const torch::Tensor& input, const torch::Tensor& parameterForward, 
        const torch::Tensor& adjOutput) const;
    std::tuple<torch::Tensor, torch::Tensor> adjointWithParameterAndFlags(
        const torch::Tensor& input, const torch::Tensor& parameterForward,
        const torch::Tensor& adjOutput, int flags) const;

    //with autograd support
    static torch::Tensor forward(c10::intrusive_ptr<EncodingBindings> self, const torch::Tensor& input);
    static torch::Tensor forwardWithParameter(c10::intrusive_ptr<EncodingBindings> self,
        const torch::Tensor& input, const torch::Tensor& parameter);
};
typedef c10::intrusive_ptr<EncodingBindings> EncodingBindings_ptr;

struct EncodingAutogradFunction : public torch::autograd::Function<EncodingAutogradFunction>
{
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        EncodingBindings_ptr activ);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs);
};
struct EncodingAutogradFunctionWithParameter : public torch::autograd::Function<EncodingAutogradFunctionWithParameter>
{
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input, const torch::Tensor& parameter,
        EncodingBindings_ptr activ);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs);
};

void bindEncoding(torch::Library& m);
