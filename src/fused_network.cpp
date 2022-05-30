#include <qmlp/fused_network.h>

#include <tinyformat.h>
#include <qmlp/qmlp.h>

QUICKMLP_NAMESPACE_BEGIN

FusedNetwork::FusedNetwork(const nlohmann::json& cfg, const std::filesystem::path& parent)
    : channelsIn_(cfg.at("num_inputs").get<int>())
    , channelsOut_(cfg.at("num_outputs").get<int>())
    , networkInputPadding_(0)
    , numParameters_(0)
{
    //kernel loader
    auto loader = QuickMLP::Instance().kernelLoader();

    //activations
    auto activationSpecs = cfg.at("activation_specification");
    auto activationFactory = ActivationFactory(activationSpecs, parent, loader);

    //encodings
    auto encodingSpecs = cfg.at("encodings");
    int encodedChannels = 0;
    if (!encodingSpecs.is_array()) throw configuration_error("Illegal type for 'encodings', array expected");
    for (const auto& cfg : encodingSpecs)
    {
        auto enc = EncodingFactory::Instance().create(cfg);
        if (enc->maxInputChannel() >= channelsIn_)
            throw configuration_error("Input channels for encoding '%s' out of bounds (num channels=%d)",
                enc->id().c_str(), channelsIn_);
        encodedChannels += enc->numOutputChannels();
        encodings_.push_back(enc);
    }

    //hidden layers
    auto networkSpecs = cfg.at("network");
    if (!cfg.is_array()) throw configuration_error("Expected a json-array for the network configuration");
    if (cfg.empty()) throw configuration_error("Empty network!");

    //loop through the layers
    int prevInput = channelsIn_;
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
                specs.activations.push_back(activationFactory.get(ae.get<std::string>()));
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
                    specs.activations.push_back(activationFactory.get(e2.get<std::string>()));
                }
            }
            else
            {
                throw configuration_error("Unknown type for the activations, expected string or array");
            }

            prevInput = specs.channelsOut;
            layers_.push_back(specs);
        }
        catch (...)
        {
            std::throw_with_nested(configuration_error("Parser error when processing network layer %d",
                static_cast<int>(layers_.size())));
        }
    }
    channelsOut_ = prevInput;

    //padding
    for (int i=0; i<layers_.size(); ++i)
    {
        int currentIn = layers_[i].channelsIn;
        int paddedIn = roundUp(currentIn, MATRIX_SIZE);
        if (currentIn == paddedIn) continue; //all is fine
        if (i==0)
        {
            networkInputPadding_ = paddedIn - currentIn;
            layers_[i].channelsIn = paddedIn;
            std::cout << "Warning: need to pad the network input channels from the encodings (" <<
                currentIn << " channels) to the next multiple of the matrix size (" <<
                MATRIX_SIZE << "), leading to a new size of " << paddedIn << std::endl;
            std::cout << "  These extra channels are padded with zeros and are thus wasted. " <<
                "Consider increasing the input encoding." << std::endl;
        } else
        {
            int padding = paddedIn - currentIn;
            layers_[i].channelsIn = paddedIn;
            layers_[i - 1].channelsOut = paddedIn;
            std::cout << "Warning: The hidden channels between layer " << (i - 1) << " and " <<
                i << " is currently specified to be " << currentIn << " channels wide." << std::endl;
            std::cout << "  Matrix multiplications, however, are always performed in multiples of " <<
                MATRIX_SIZE << " channels." << std::endl;
            std::cout << "  The network has been automatically increased in size to fit " <<
                paddedIn << " channels. Consider updating the network specification to reflect this." << std::endl;
        }
    }

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

int FusedNetwork::networkParameterCount() const
{
    return numParameters_;
}

Tensor::Precision FusedNetwork::networkParameterPrecision(Tensor::Usage usage) const
{
    switch (usage)
    {
    case Tensor::INFERENCE: return Tensor::HALF;
    case Tensor::GRADIENTS: return Tensor::FLOAT;
    default: throw std::runtime_error("Unknown usage");
    }
}

void FusedNetwork::setNetworkParameter(const Tensor& tensor, Tensor::Usage usage)
{
    if (tensor.ndim() != 1 || tensor.sizes()[0] != numParameters_)
    {
        throw std::runtime_error(tinyformat::format(
            "Illegal tensor shape, 1D tensor with %d entries required", numParameters_));
    }
    if (tensor.precision() != networkParameterPrecision(usage))
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

void FusedNetwork::inference(const Tensor& input, Tensor& output, CUstream stream)
{
    throw std::logic_error("Not implemented");
}


QUICKMLP_NAMESPACE_END
