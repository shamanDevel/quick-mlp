#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>
#include <mutex>

#include <qmlp/fused_network.h>
#include <qmlp/qmlp.h>
#include "../qmlp/bindings.h"
#include "bind_encoding.h"

class NetworkBindings : public torch::CustomClassHolder
{
    std::string cfg_;
    std::string parent_;
    std::mutex mutex_;
    QUICKMLP_NAMESPACE::FusedNetwork_ptr n_;
    friend class NetworkAutogradFunction;

public:
    NetworkBindings(const std::string& cfg, const std::string& parent)
        : cfg_(cfg), parent_(parent), n_()
    {
        nlohmann::json j = nlohmann::json::parse(cfg);
        n_ = std::make_shared<QUICKMLP_NAMESPACE::FusedNetwork>(cfg, parent);
    }

    [[nodiscard]] std::string cfg() const
    {
        return cfg_;
    }
    [[nodiscard]] std::string parent() const
    {
        return parent_;
    }

    [[nodiscard]] static int64_t MatrixSize() { return QUICKMLP_NAMESPACE::FusedNetwork::MATRIX_SIZE; }
    [[nodiscard]] static int64_t WarpSize() { return QUICKMLP_NAMESPACE::FusedNetwork::WARP_SIZE; }
    [[nodiscard]] static int64_t MaxSharedMemoryBytes() { return QUICKMLP_NAMESPACE::FusedNetwork::MAX_SHARED_MEMORY_BYTES; }

    [[nodiscard]] bool isParallelStreams() const { return n_->isParallelStreams(); }
    void setParallelStreams(bool enabled) { n_->setParallelStreams(enabled); }

    [[nodiscard]] int64_t numEncodings() const { return n_->numEncodings(); }
    EncodingBindings_ptr encoding(int64_t idx);

    [[nodiscard]] int64_t channelsIn() const { return n_->channelsIn(); }
    [[nodiscard]] int64_t channelsOut() const { return n_->channelsOut(); }
    [[nodiscard]] int64_t numLayers() const { return n_->numLayers(); }
    [[nodiscard]] int64_t networkParameterCount() const { return n_->networkParameterCount(); }

    /**
     * Creates an empty parameter fit for storing the inference parameters.
     * Note that the memory is uninitialized!
     */
    [[nodiscard]] torch::Tensor createInferenceParameters() const
    {
        return torch::empty({ static_cast<long long>(n_->networkParameterCount()) },
            at::TensorOptions().dtype(
            n_->networkParameterPrecision(qmlp::Tensor::INFERENCE) == qmlp::Tensor::FLOAT ? c10::kFloat : c10::kHalf
        ).device(c10::kCUDA).requires_grad(true));
    }
    /**
     * Creates an empty parameter fit for storing the gradient parameters.
     * Note that the memory is uninitialized!
     */
    [[nodiscard]] torch::Tensor createGradientParameters() const
    {
        return torch::empty({ static_cast<long long>(n_->networkParameterCount()) },
            at::TensorOptions().dtype(
                n_->networkParameterPrecision(qmlp::Tensor::GRADIENTS) == qmlp::Tensor::FLOAT ? c10::kFloat : c10::kHalf
            ).device(c10::kCUDA));
    }

    /**
     * Returns a view of the inference or the gradient tensor
     * slicing the weight matrix or bias vector
     */
    [[nodiscard]] torch::Tensor view(int64_t layer, bool bias, torch::Tensor parameters);

    void initializeInferenceParameters(torch::Tensor dst);

    //no gradients
    torch::Tensor inference(
        const torch::Tensor& input, 
        const torch::Tensor& networkParameters,
        const std::vector<torch::Tensor>& encodingParameters);

    //with autograd support
    torch::Tensor forward(
        const torch::Tensor& input,
        const torch::Tensor& networkParameters,
        const std::vector<torch::Tensor>& encodingParameters);

private:
    //note: hold lock guard during setting parameters and running the network!
    void setNetworkParameters(const torch::Tensor& networkParameters,
        const std::vector<torch::Tensor>& encodingParameters);
};
typedef c10::intrusive_ptr<NetworkBindings> NetworkBindings_ptr;

void bindNetwork(torch::Library& m);
