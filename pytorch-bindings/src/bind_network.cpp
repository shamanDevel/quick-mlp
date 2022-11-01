#include "bind_network.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <limits>
#include "tensorlist_node.h"

static torch::Tensor createOutputTensor(const torch::Tensor& input, int channels)
{
    TORCH_CHECK(input.dim() == 2, "Inputs must be 2D (B,C)");
    auto B = input.size(0);
    auto t = torch::empty({ B, channels }, at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA).memory_format(c10::MemoryFormat::Contiguous));
    TORCH_CHECK(t.stride(1) == 1);
    return t;
}

EncodingBindings_ptr NetworkBindings::encoding(int idx)
{
    return c10::make_intrusive<EncodingBindings>(n_->encoding(idx));
}

void NetworkBindings::initializeInferenceParameters(torch::Tensor dst)
{
    std::lock_guard lock(mutex_);
    n_->setNetworkParameter(QUICKMLP_NAMESPACE::wrap(dst), qmlp::Tensor::INFERENCE);
    int64_t seed = *torch::randint(
        std::numeric_limits<int64_t>::max(),
        at::IntArrayRef{ 1 },
        at::TensorOptions().dtype(c10::kLong).device(c10::kCPU))
        .data<int64_t>();
    std::default_random_engine rng(seed);
    n_->initializeInferenceParameters(rng);
}

void NetworkBindings::setNetworkParameters(const torch::Tensor& networkParameters,
    const std::vector<torch::Tensor>& encodingParameters)
{
    if (networkParameters.defined()) {
        n_->setNetworkParameter(QUICKMLP_NAMESPACE::wrap(networkParameters),
            qmlp::Tensor::INFERENCE);
    }

    int m = std::min(static_cast<int>(encodingParameters.size()),
        n_->numEncodings());
    for (int i = 0; i < m; ++i)
    {
        if (encodingParameters[i].defined())
            n_->encoding(i)->setParameter(
                QUICKMLP_NAMESPACE::wrap(encodingParameters[i]),
                qmlp::Tensor::INFERENCE);
    }
}

torch::Tensor NetworkBindings::inference(const torch::Tensor& input, 
                                         const torch::Tensor& networkParameters,
                                         const std::vector<torch::Tensor>& encodingParameters)
{
    std::lock_guard lock(mutex_);

    TORCH_CHECK(input.defined());
    torch::Tensor output = createOutputTensor(input, n_->channelsOut());

    setNetworkParameters(networkParameters, encodingParameters);

    CUstream stream = c10::cuda::getCurrentCUDAStream();
    auto outputWrapped = QUICKMLP_NAMESPACE::wrap(output);
    n_->inference(
        QUICKMLP_NAMESPACE::wrap(input),
        outputWrapped,
        stream);

    return output;
}

torch::Tensor NetworkBindings::forward(
    const torch::Tensor& input, const torch::Tensor& networkParameters,
    const std::vector<torch::Tensor>& encodingParameters)
{
    TORCH_CHECK(input.defined());

    //TODO: Autograd API does not support lists of tensors as input.
    //Write a custom Node here
    //see https://discuss.pytorch.org/t/extending-autograd-from-c/76240/5

    //check adjoint mode
    QUICKMLP_NAMESPACE::FusedNetwork::AdjointModeFlags adjointFlags = 0;
    if (input.requires_grad())
        adjointFlags |= QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_INPUT;
    if (networkParameters.defined() && networkParameters.requires_grad())
        adjointFlags |= QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_NETWORK_WEIGHTS;
    for (const auto& t : encodingParameters)
    {
        if (t.defined() && t.requires_grad())
            adjointFlags |= QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_INPUT_ENCODINGS;
    }

    //collect variables
    torch::autograd::variable_list vars(3 + encodingParameters.size());
    vars[0] = input;
    vars[1] = networkParameters;
    for (size_t i = 0; i < encodingParameters.size(); ++i)
        vars[i + +3] = networkParameters[i];

    torch::autograd::tensor_list ret = TensorlistFunction::apply(
        vars, 
        [this, adjointFlags]
        (TensorlistAutogradContext* ctx, torch::autograd::variable_list args) -> torch::autograd::variable_list
        {
            //forward
            const auto& input = args[0];
            const auto& networkParameters = args[1];
            std::vector<at::Tensor> encodingParameters(args.begin() + 3, args.end());

            long long forwardMemorySize = this->n_->forwardMemory(
                input.size(0),
                static_cast<QUICKMLP_NAMESPACE::FusedNetwork::AdjointModeFlags>(adjointFlags));
            torch::Tensor tmpForward = torch::empty(
                { forwardMemorySize },
                at::TensorOptions().dtype(c10::kByte).device(input.device()));
            args[2] = tmpForward;

            torch::Tensor output = createOutputTensor(input, this->n_->channelsOut());
            CUstream stream = c10::cuda::getCurrentCUDAStream();
            auto outputWrapped = QUICKMLP_NAMESPACE::wrap(output);

            std::lock_guard lock(this->mutex_);
            this->setNetworkParameters(networkParameters, encodingParameters);
            this->n_->forward(
                QUICKMLP_NAMESPACE::wrap(input),
                outputWrapped,
                tmpForward.data_ptr<char>(),
                stream);

            ctx->save_for_backward(args);

            return { output };
        },
            [this, adjointFlags]
        (TensorlistAutogradContext* ctx, torch::autograd::variable_list grad_outputs) -> torch::autograd::variable_list
        {
            //adjoint
            auto saved = ctx->get_saved_variables();
            torch::Tensor input = saved[0];
            torch::Tensor networkParameters = saved[1];
            torch::Tensor forwardTmp = saved[2];
            std::vector<torch::Tensor> encodingParameters(saved.begin() + 3, saved.end());

            torch::Tensor adjOutput = grad_outputs[0];

            std::lock_guard lock(this->mutex_);
            this->setNetworkParameters(networkParameters, encodingParameters);

            torch::Tensor adjInput;
            if (adjointFlags & QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_INPUT)
                adjInput = torch::zeros_like(input);
            auto adjInputWrapped = QUICKMLP_NAMESPACE::wrap(adjInput);

            torch::Tensor adjNetworkParameters;
            if (adjointFlags & QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_NETWORK_WEIGHTS)
            {
                adjNetworkParameters = torch::zeros_like(networkParameters);
                this->n_->setNetworkParameter(
                    QUICKMLP_NAMESPACE::wrap(adjNetworkParameters),
                    QUICKMLP_NAMESPACE::Tensor::GRADIENTS);
            }

            std::vector<torch::Tensor> adjEncodingParameters(encodingParameters.size());
            int m = std::min(static_cast<int>(encodingParameters.size()),
                this->n_->numEncodings());
            if (adjointFlags & QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_INPUT_ENCODINGS)
            {
                for (int i = 0; i < m; ++i)
                {
                    if (encodingParameters[i].defined()) {
                        adjEncodingParameters[i] = torch::zeros_like(encodingParameters[i]);
                        this->n_->encoding(i)->setParameter(
                            QUICKMLP_NAMESPACE::wrap(adjEncodingParameters[i]),
                            qmlp::Tensor::GRADIENTS);
                    }
                }
            }

            long long adjointMemorySize = this->n_->adjointMemory(
                input.size(0),
                adjointFlags);
            torch::Tensor adjointTmp = torch::empty(
                { adjointMemorySize },
                at::TensorOptions().dtype(c10::kByte).device(input.device()));

            CUstream stream = c10::cuda::getCurrentCUDAStream();
            this->n_->adjoint(
                QUICKMLP_NAMESPACE::wrap(input),
                QUICKMLP_NAMESPACE::wrap(adjOutput),
                adjointFlags,
                adjInputWrapped,
                forwardTmp.data_ptr<char>(),
                adjointTmp.data_ptr<char>(),
                stream);

            torch::autograd::variable_list adjInputs(saved.size());
            adjInputs[0] = adjInput;
            adjInputs[1] = adjNetworkParameters;
            for (int i = 0; i < m; ++i)
                adjInputs[i + 3] = adjEncodingParameters[i];

            return adjInputs;
        });
    return ret[0];
}


void bindNetwork(torch::Library& m)
{
    m.class_<NetworkBindings>("Network")
        .def(torch::init<std::string, std::string>(), R"doc(
Constructs a new fused network from the given json configuration. 
The second parameter denotes the parent folder of the config, from this folder,
linked activation specifications are loaded.)doc", 
            { torch::arg("cfg"), torch::arg("parent") = "."})
        .def("cfg", &NetworkBindings::cfg, "Returns the Json configuration used to construct this network")
        .def("parent", &NetworkBindings::parent, "The parent folder of the configuration, used to load activation definitions")
        .def_static("MatrixSize", &NetworkBindings::MatrixSize)
        .def_static("WarpSize", &NetworkBindings::WarpSize)
        .def_static("MaxSharedMemoryBytes", &NetworkBindings::MaxSharedMemoryBytes)
        .def("is_parallel_streams", &NetworkBindings::isParallelStreams,
            "True iff parallel streams during backpropagation are enabled")
        .def("set_parallel_streams", &NetworkBindings::setParallelStreams,
            "Enables or disables parallel streams during backpropagation. Might speed up the computation")
        .def("num_encodings", &NetworkBindings::numEncodings, 
            "Returns the number of encodings prepending the multilayer perceptron")
        .def("encoding", &NetworkBindings::encoding, R"doc(
Returns the encoding at the given index.
Use this to access and modify the state of that encoding.)doc")
        .def("channels_in", &NetworkBindings::channelsIn,
            "The expected input channel count")
        .def("channels_out", &NetworkBindings::channelsOut,
            "The output channel count produced by the network")
        //TODO: continue with the bindings
        .def_pickle(
            [](const c10::intrusive_ptr<NetworkBindings>& self) -> std::string {
                return self->cfg();
            },
            [](std::string state) -> c10::intrusive_ptr<NetworkBindings> {
                return c10::make_intrusive<NetworkBindings>(std::move(state));
            })
        ;
}

