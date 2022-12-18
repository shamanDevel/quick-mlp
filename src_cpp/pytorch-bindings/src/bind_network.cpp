#include "bind_network.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <sstream>
#include <spdlog/spdlog.h>

#include <limits>
#include "tensorlist_node.h"


// Formatter specializations for debugging / logging
template <> struct fmt::formatter<c10::ArrayRef<int64_t>> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(const c10::ArrayRef<int64_t>& c, FormatContext& ctx) const {
        std::stringstream ss;
        ss << c;
        return fmt::formatter<std::string>::format(ss.str(), ctx);
    }
};
template <> struct fmt::formatter<caffe2::TypeMeta> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(const caffe2::TypeMeta& c, FormatContext& ctx) const {
        std::stringstream ss;
        ss << c;
        return fmt::formatter<std::string>::format(ss.str(), ctx);
    }
};
template <> struct fmt::formatter<c10::Device> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(const c10::Device& c, FormatContext& ctx) const {
        std::stringstream ss;
        ss << c;
        return fmt::formatter<std::string>::format(ss.str(), ctx);
    }
};


static torch::Tensor createOutputTensor(const torch::Tensor& input, int channels)
{
    TORCH_CHECK(input.dim() == 2, "Inputs must be 2D (B,C), but has shape ", input.sizes());
    auto B = input.size(0);
    auto t = torch::empty({ B, channels }, at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA).memory_format(c10::MemoryFormat::Contiguous));
    TORCH_CHECK(t.stride(1) == 1);
    return t;
}

NetworkBindings::NetworkBindings(const std::string& cfg, const std::string& parent): cfg_(cfg), parent_(parent), n_()
{
    nlohmann::json j = nlohmann::json::parse(cfg);
    n_ = std::make_shared<QUICKMLP_NAMESPACE::FusedNetwork>(j, parent);
}

EncodingBindings_ptr NetworkBindings::encoding(int64_t idx)
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

    auto np = networkParameters.dtype() == c10::kHalf
        ? networkParameters
        : networkParameters.to(c10::kHalf);
    setNetworkParameters(np, encodingParameters);

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
    auto logger = qmlp::QuickMLP::Instance().getLogger();
    logger->debug("Input: shape={}, dtype={}, device={}",
        input.sizes(), input.dtype(), input.device());
    logger->debug("Network parameters: shape={}, dtype={}, device={}",
        networkParameters.sizes(), networkParameters.dtype(), networkParameters.device());
    for (size_t i=0; i<encodingParameters.size(); ++i)
    {
        if (encodingParameters[i].defined())
        {
            logger->debug("Encoding parameter {}: shape={}, dtype={}, device={}",
                i, encodingParameters[i].sizes(), encodingParameters[i].dtype(),
                encodingParameters[i].device());
        }
        else
        {
            logger->debug("Encoding parameter {} is undefined", i);
        }
    }

    TORCH_CHECK(input.defined());

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
    logger->debug("Adjoint flags: {}", int(adjointFlags));
    //short-cut
    if (adjointFlags == 0)
    {
        //no derivatives activate
        return inference(input, networkParameters, encodingParameters);
    }

    //collect variables
    //I can't store None tensors, so compact the list here
    torch::autograd::variable_list vars;
    int parameterIndexIntoVars = -1;
    std::vector<int> encodingIndexIntoVars(numEncodings(), -1);
    vars.push_back(input);
    caffe2::TypeMeta originalNetworkParamterDtype;
    if (networkParameters.defined())
    {
        originalNetworkParamterDtype = networkParameters.dtype();
        parameterIndexIntoVars = vars.size();
        auto np = networkParameters.dtype() == c10::kHalf
            ? networkParameters
            : networkParameters.to(c10::kHalf);
        vars.push_back(np);
    }
    for (size_t i=0; i<encodingParameters.size(); ++i)
    {
        if (encodingParameters[i].defined())
        {
            encodingIndexIntoVars[i] = vars.size();
            vars.push_back(encodingParameters[i]);
        }
    }

    torch::autograd::tensor_list ret = TensorlistFunction::apply(
        vars, 
        [this, logger, adjointFlags, parameterIndexIntoVars, encodingIndexIntoVars, originalNetworkParamterDtype]
        (TensorlistAutogradContext* ctx, torch::autograd::variable_list args) -> torch::autograd::variable_list
        {
            //forward
            const auto input = args[0];
            const auto networkParameters =
                parameterIndexIntoVars >= 0
                ? args[parameterIndexIntoVars]
                : torch::autograd::Variable();
            std::vector<torch::autograd::Variable> encodingParameters(this->numEncodings());
            for (int i=0; i<this->numEncodings(); ++i)
            {
                if (encodingIndexIntoVars[i] >= 0)
                    encodingParameters[i] = args[encodingIndexIntoVars[i]];
            }

            long long forwardMemorySize = this->n_->forwardMemory(
                input.size(0),
                adjointFlags);
            logger->debug("Allocate forward memory with {} bytes", forwardMemorySize);
            torch::Tensor tmpForward = torch::empty(
                { forwardMemorySize },
                at::TensorOptions().dtype(c10::kByte).device(input.device()));
            args.push_back(tmpForward);

            torch::Tensor output = createOutputTensor(input, this->n_->channelsOut());
            CUstream stream = c10::cuda::getCurrentCUDAStream();
            auto outputWrapped = QUICKMLP_NAMESPACE::wrap(output);

            std::lock_guard lock(this->mutex_);

            logger->debug("Set network and encoding parameters");
            this->setNetworkParameters(networkParameters, encodingParameters);

            logger->debug("Launch forward kernel");
            this->n_->forward(
                QUICKMLP_NAMESPACE::wrap(input),
                outputWrapped,
                tmpForward.data_ptr(),
                stream);

            ctx->save_for_backward(args);

            return { output };
        },
        [this, logger, adjointFlags, parameterIndexIntoVars, encodingIndexIntoVars, originalNetworkParamterDtype]
        (TensorlistAutogradContext* ctx, torch::autograd::variable_list grad_outputs) -> torch::autograd::variable_list
        {
            //adjoint
            auto saved = ctx->get_saved_variables();
            torch::Tensor input = saved[0];
            const auto networkParameters =
                parameterIndexIntoVars >= 0
                ? saved[parameterIndexIntoVars]
                : torch::autograd::Variable();
            std::vector<torch::autograd::Variable> encodingParameters(this->numEncodings());
            for (int i = 0; i < this->numEncodings(); ++i)
            {
                if (encodingIndexIntoVars[i] >= 0)
                    encodingParameters[i] = saved[encodingIndexIntoVars[i]];
            }
            torch::Tensor forwardTmp = saved.back();

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
                adjNetworkParameters = torch::zeros(networkParameters.sizes(),
                    networkParameters.options().dtype(unwrapDtype(
                        this->n_->networkParameterPrecision(qmlp::Tensor::GRADIENTS))));
                this->n_->setNetworkParameter(
                    QUICKMLP_NAMESPACE::wrap(adjNetworkParameters),
                    QUICKMLP_NAMESPACE::Tensor::GRADIENTS);
            }

            std::vector<torch::Tensor> adjEncodingParameters(encodingParameters.size());
            int m = this->n_->numEncodings();
            if (adjointFlags & QUICKMLP_NAMESPACE::FusedNetwork::AdjointMode::GRADIENTS_INPUT_ENCODINGS)
            {
                for (int i = 0; i < m; ++i)
                {
                    if (encodingParameters[i].defined()) {
                        adjEncodingParameters[i] = torch::zeros(adjEncodingParameters[i].sizes(),
                            adjEncodingParameters[i].options().dtype(unwrapDtype(
                                this->n_->encoding(i)->parameterPrecision(qmlp::Tensor::GRADIENTS))));
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
                forwardTmp.data_ptr(),
                adjointTmp.data_ptr(),
                stream);

            torch::autograd::variable_list adjInputs(saved.size());
            adjInputs[0] = adjInput;
            if (parameterIndexIntoVars >= 0)
                adjInputs[parameterIndexIntoVars] = adjNetworkParameters.to(originalNetworkParamterDtype);
            for (int i = 0; i < m; ++i) {
                if (encodingIndexIntoVars[i] >= 0)
                    adjInputs[encodingIndexIntoVars[i]] = adjEncodingParameters[i];
            }

            return adjInputs;
        });
    return ret[0];
}

torch::Tensor NetworkBindings::view(int64_t layer, bool bias, torch::Tensor parameters)
{
    TORCH_CHECK(parameters.dtype() == c10::kFloat || parameters.dtype() == c10::kHalf);
    TORCH_CHECK(parameters.dim() == 1, "The tensor must be of shape (num_parameters, )");
    TORCH_CHECK(parameters.size(0) == n_->networkParameterCount());

    QUICKMLP_NAMESPACE::Tensor::Precision p;
    void* ptr;
    if (parameters.dtype() == c10::kFloat)
    {
        p = qmlp::Tensor::FLOAT;
        ptr = parameters.data_ptr();
    }
    else
    {
        p = qmlp::Tensor::HALF;
        ptr = parameters.data_ptr();
    }

    auto v = n_->networkParameter(layer, bias, ptr, p);
    return QUICKMLP_NAMESPACE::unwrap(v);
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
        .def("num_layers", &NetworkBindings::numLayers,
            "The number of layers in the network")
        .def("num_network_parameters", &NetworkBindings::networkParameterCount,
            "The number of parameters in the network (size of the parameter tensor)")
        .def("create_inference_parameters", &NetworkBindings::createInferenceParameters,
            "Creates the tensor storing the inference parameters. Dtype=float16, Shape=(num_network_parameters(),).\nNote: the memory is uninitialized!")
        //  not needed in the public API:
        //.def("create_gradient_parameters", &NetworkBindings::createGradientParameters,
        //    "Creates the tensor storing the gradients. Dtype=float32, Shape=(num_network_parameters(),)")
        .def("initialize_inference_parameters", &NetworkBindings::initializeInferenceParameters,
            "Default initialization of the inference parameters")
        .def("parameter_view", &NetworkBindings::view,
            "Returns a view into the inference or the gradient tensor. The returning tensor allows read-write access to the weight and bias matrix at the given layer.")
        .def("inference", &NetworkBindings::inference, R"doc(
Performs inference, no intermediate states are recorded and differentiation is not possible.
In most cases, you rather want to call 'forward', as this handles all cases (with and without gradients).
:param input: The input positions, shape=(batch, dimensions), dtype=float32
:param network_parameters: The network parameters. 1D tensor with self.num_network_parameters() elements and dtype=float16.
   See also self.create_inference_parameters() for an utility that creates a tensor of the correct type.
:param encoding_parameters: a vector with as many entries as there are input encodings.
   For every entry, contains the parameters of that encoding or None if that encoding does not require any.
:returns: The output predictions of shape (batch, out-dimensions), dtype=float32
)doc", {torch::arg("input"), torch::arg("network_parameters"),
            torch::arg("encoding_parameters")=std::vector<torch::Tensor>()})
        .def("forward", &NetworkBindings::forward, R"doc(
Performs a forward pass. This function is fully differentiable and can be seamlessly 
integrated in the computation graph.
:param input: The input positions, shape=(batch, dimensions), dtype=float32
:param network_parameters: The network parameters. 1D tensor with self.num_network_parameters() elements and dtype=float16.
   See also self.create_inference_parameters() for an utility that creates a tensor of the correct type.
:param encoding_parameters: a vector with as many entries as there are input encodings.
   For every entry, contains the parameters of that encoding or None if that encoding does not require any.
:returns: The output predictions of shape (batch, out-dimensions), dtype=float32
)doc", { torch::arg("input"), torch::arg("network_parameters"),
            torch::arg("encoding_parameters") = std::vector<torch::Tensor>() })
        .def_pickle(
            [](const c10::intrusive_ptr<NetworkBindings>& self) -> std::vector<std::string> {
                return { self->cfg(), self->parent() };
            },
            [](std::vector<std::string> state) -> c10::intrusive_ptr<NetworkBindings> {
                TORCH_CHECK(state.size() == 2, "Corrupt state");
                return c10::make_intrusive<NetworkBindings>(std::move(state[0]), std::move(state[1]));
            })
        ;
}

