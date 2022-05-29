#include <qmlp/inner_network.h>

#include <tinyformat.h>

QUICKMLP_NAMESPACE_BEGIN

InnerNetwork::InnerNetwork(const nlohmann::json& cfg, int inputChannels, ActivationFactory_ptr activations)
    : channelsIn_(inputChannels)
    , channelsOut_(0)
    , numParameters_(0)
{
    if (!cfg.is_array()) throw configuration_error("Expected a json-array for the network configuration");
    if (cfg.empty()) throw configuration_error("Empty network!");

    //loop through the layers
    int prevInput = inputChannels;
    for (const auto& e : cfg)
    {
        try
        {
            //base parameters
            LayerSpecification specs;
            specs.channelsIn = prevInput;
            specs.channelsOut = e.at("n_out").get<int>();
            specs.useBias = e.at("bias").get<bool>();

            //either a single activation for all channels or one activation per channel
            auto ae = e.at("activation");
            if (ae.is_string())
            {
                specs.activations.push_back(activations->get(ae.get<std::string>()));
            }
            else if (ae.is_array())
            {
                if (ae.size() != specs.channelsOut)
                {
                    throw configuration_error(
                        "If you specify individual activations per channel, then the number of activations (%d) must match the number of output channels (%d)",
                        static_cast<int>(ae.size()), specs.channelsOut);
                }
                for (const auto& e2 : ae)
                {
                    if (!e2.is_string()) throw configuration_error("Per-channel activations must be specified as a string");
                    specs.activations.push_back(activations->get(e2.get<std::string>()));
                }
            }
            else
            {
                throw configuration_error("Unknown type for the activations, expected string or array");
            }

            prevInput = specs.channelsOut;
            layers_.push_back(specs);
        } catch (...)
        {
            std::throw_with_nested(configuration_error("Parser error when processing network layer %d",
                static_cast<int>(layers_.size())));
        }
    }
    channelsOut_ = prevInput;

    //prefix sum for the offsets of weight + bias
    //First are all weights, then all biases
    for (auto& specs : layers_)
    {
        specs.weightsStart = numParameters_;
        numParameters_ += specs.channelsIn * specs.channelsOut;
    }
    for (auto& specs : layers_)
    {
        specs.biasStart = numParameters_;
        numParameters_ += specs.channelsOut;
    }
}

int InnerNetwork::parameterCount() const
{
    return numParameters_;
}

Tensor::Precision InnerNetwork::parameterPrecision(Tensor::Usage usage) const
{
    switch (usage)
    {
    case Tensor::INFERENCE: return Tensor::HALF;
    case Tensor::GRADIENTS: return Tensor::FLOAT;
    default: throw std::runtime_error("Unknown usage");
    }
}

void InnerNetwork::setParameter(const Tensor& tensor, Tensor::Usage usage)
{
    if (tensor.ndim() != 1 || tensor.sizes()[0] != numParameters_)
    {
        throw std::runtime_error(tinyformat::format(
            "Illegal tensor shape, 1D tensor with %d entries required", numParameters_));
    }
    if (tensor.precision() != parameterPrecision(usage))
    {
        throw std::runtime_error("Wrong data type for the parameter. See parameterPrecision() for the expected format");
    }

    switch (usage)
    {
    case Tensor::INFERENCE:
        parametersInference_ = tensor;
        break;
    case Tensor::GRADIENTS:
        parametersGradients_ = tensor;
        break;
    }
}

void InnerNetwork::fillParameterConstant(const std::string& constantName, const ckl::KernelFunction& function,
    CUstream stream)
{
    throw std::logic_error("Not implemented");
}

QUICKMLP_NAMESPACE_END
