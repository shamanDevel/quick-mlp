#include <qmlp/fused_network.h>

#include <tinyformat.h>
#include <qmlp/qmlp.h>
#include <unordered_set>
#include <qmlp/kernels/tensor.cuh>

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
    if (!networkSpecs.is_array()) throw configuration_error("Expected a json-array for the network configuration");
    if (networkSpecs.empty()) throw configuration_error("Empty network!");

    //loop through the layers
    int prevInput = channelsIn_;
    for (const auto& e : networkSpecs)
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
                MATRIX_SIZE << "), leading to a new size of " << paddedIn << ". ";
            std::cout << "These extra channels are padded with zeros and are thus wasted. " <<
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
    //TODO: for now, also pad the last layer
    //In the future, add special handling for small last layers
    {
        int currentOut = layers_.rbegin()->channelsOut;
        int paddedOut = roundUp(currentOut, MATRIX_SIZE);
        if (currentOut != paddedOut)
        {
            layers_.rbegin()->channelsOut = paddedOut;
            std::cout << "Warning: the last layer's output is not a multiple of " << MATRIX_SIZE <<
                ", padding required. In the future, I will add special handling for the last layer." << std::endl;
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

IEncoding_ptr FusedNetwork::encoding(int idx)
{
    if (idx < 0 || idx >= encodings_.size())
        throw std::runtime_error("Array index out of bounds");
    return encodings_[idx];
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

Tensor FusedNetwork::networkParameter(int layer, bool bias, Tensor::Usage usage)
{
    if (layer < 0 || layer >= layers_.size())
        throw std::runtime_error("Layer index out of bounds");
    const auto& l = layers_[layer];

    auto& p = usage == Tensor::INFERENCE ? parametersInference_ : parametersGradients_;
    int8_t* data = static_cast<int8_t*>(p.rawPtr());
    int bytesPerEntry = Tensor::BytesPerEntry[p.precision()];
    if (bias)
    {
        return Tensor(
            data + bytesPerEntry * l.biasStart,
            p.precision(),
            { l.channelsOut }, { 1 }
        );
    }
    else
    {
        return Tensor(
            data + bytesPerEntry * l.weightsStart,
            p.precision(),
            { l.channelsOut, l.channelsIn }, { l.channelsIn, 1 }
        );
    }
}

static void replaceAll(std::string& s, const std::string& search, const std::string& replace) {
    for (size_t pos = 0; ; pos += replace.length()) {
        // Locate the substring to replace
        pos = s.find(search, pos);
        if (pos == std::string::npos) break;
        // Replace by erasing and inserting
        //TODO: might be more efficient by overwriting the character positions that are shared
        // and only erasing/inserting the difference
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

void FusedNetwork::inference(const Tensor& input, Tensor& output, CUstream stream)
{
    CHECK_DIM(input, 2);
    CHECK_DIM(output, 2);
    CHECK_SIZE(input, 1, channelsIn());
    CHECK_SIZE(output, 1, channelsOut());
    int numel = input.size(0);
    CHECK_SIZE(output, 0, numel);
    CHECK_DTYPE(input, precisionIn());
    CHECK_DTYPE(output, precisionOut());

    auto kl = QuickMLP::Instance().kernelLoader();
    std::string codeTemplate = kl->findFile("qmlp/kernels/network.cuh").value();

    //GENERATE CODE
    //encodings
    std::stringstream encodingIncludes;
    std::stringstream encodingConstants;
    std::vector<std::string> encodingConstantNames;
    const auto constantNameForEncoding = [](IEncoding_ptr e, int encodingIdx)
    {
        return tinyformat::format("cEncoding%d", encodingIdx);
    };
    for (int encodingIdx=0; encodingIdx<encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        e->fillCode(encodingIncludes);
        encodingIncludes << "\n";
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            encodingConstants << "__constant__ " << e->parameterName() << constantName << "\n";
            encodingConstantNames.push_back(constantName);
        }
    }
    replaceAll(codeTemplate, "$$INCLUDES$$", encodingIncludes.str());
    replaceAll(codeTemplate, "$$ENCODING_CONSTANTS$$", encodingConstants.str());

    //activations
    std::stringstream activationDefinitions;
    std::unordered_set<std::string> activationIDs;
    for (const auto& l : layers_)
    {
        for (const auto& a : l.activations)
        {
            if (activationIDs.count(a->id())==0)
            {
                activationDefinitions << a->code() << "\n";
                activationIDs.insert(a->id());
            }
        }
    }
    replaceAll(codeTemplate, "$$DEFINE_ACTIVATIONS$$", activationDefinitions.str());

    //constants
    int maxChannels = 0;
    for (const auto& l : layers_)
        maxChannels = std::max({ maxChannels, l.channelsIn, l.channelsOut });
    int inputPadStart = 0;
    for (const auto& e : encodings_)
        inputPadStart = std::max(inputPadStart, e->maxInputChannel() + 1);
    int channelsIn = layers_.begin()->channelsIn;
    int channelsOut = layers_.rbegin()->channelsOut;
    replaceAll(codeTemplate, "$$MAX_CHANNELS$", std::to_string(maxChannels));
    replaceAll(codeTemplate, "$$INPUT_PAD_START$$", std::to_string(inputPadStart));
    replaceAll(codeTemplate, "$$CHANNELS_IN$$", std::to_string(channelsIn));
    replaceAll(codeTemplate, "$$CHANNELS_OUT$$", std::to_string(channelsOut));

    //call encodings
    std::stringstream callEncodings;
    int encodingOffset = 0;
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        callEncodings << e->qualifiedName() << "::forward(encodingInput, encodingOutput + " <<
            encodingOffset;
        encodingOffset += e->numOutputChannels();
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            callEncodings << ", " << constantName;
        }
        callEncodings << ");\n";
    }
    replaceAll(codeTemplate, "$$CALL_ENCODINGS$$", callEncodings.str());

    //call layers
    //TODO: prefetch weights in shared memory?
    std::stringstream callLayers;
    for (const auto& l : layers_)
    {
        //type
        if (l.activations.size() > 1) throw std::runtime_error("Individual activations per channel are not implemented yet");
        callLayers << "qmlp::kernel::Layer<" << (l.channelsIn / 16) << ", " << (l.channelsOut / 16) <<
            ", " << maxChannels << ", " << (l.useBias ? "true" : "false") << ", " <<
            "activations::" << l.activations[0]->id() << ">";
        //parameters
        callLayers << "::inference(networkParameters+" << l.weightsStart << ", ";
        if (l.useBias)
            callLayers << "networkParameters+" << l.biasStart << ", ";
        else
            callLayers << "nullptr, ";
        callLayers << "intermediateResultsWarp);\n";
    }
    replaceAll(codeTemplate, "$$CALL_NETWORK_LAYERS$$", callLayers.str());

    //COMPILE
    int compileFlags = ckl::KernelLoader::CompilationFlags::CompileThrowOnError;
#ifndef NDEBUG
    compileFlags |= ckl::KernelLoader::CompilationFlags::CompileDebugMode
        | ckl::KernelLoader::CompilationFlags::CompileVerboseLogging;
#endif
    ckl::KernelFunction fun = kl->getKernel(
        "qmlp::kernel::NetworkKernelInference",
        codeTemplate,
        encodingConstantNames,
        compileFlags).value();

    //CONSTANTS
    //fill constants of the encodings
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            e->fillParameterConstant(constantName, fun, stream);
        }
    }

    //LAUNCH KERNEL
    //compute shared memory
    int bestBlockSize = fun.bestBlockSize();
    int blockSize = bestBlockSize;
    int sharedMemorySize = bestBlockSize * maxChannels * sizeof(half);
    if (sharedMemorySize>MAX_SHARED_MEMORY_BYTES)
    {
        blockSize = MAX_SHARED_MEMORY_BYTES / (maxChannels * sizeof(half));
        blockSize = (blockSize / 32) * 32; //round up to multiple of the warp size
        std::cout << "It would be possible to fit more threads into each block in terms of register usage, but the shared memory is not enough. " <<
            "Reducing the block size from " << bestBlockSize << " down to " << blockSize << std::endl;
        sharedMemorySize = blockSize * maxChannels * sizeof(half);
    }
    int minGridSize = std::min(
        CKL_DIV_UP(numel, blockSize), 
        fun.minGridSize());
    std::cout << "Launch with a block size of " << blockSize << " and a shared memory size of " <<
        sharedMemorySize << std::endl;
    //launch
    auto inputAcc = input.accessor<kernel::Tensor2Read<float>>();
    auto outputAcc = input.accessor<kernel::Tensor2RW<float>>();
    const half* networkParams = parametersInference_.dataPtr<half>();
    fun.call(minGridSize, blockSize, sharedMemorySize, stream, 
        numel, inputAcc, outputAcc, networkParams);
}


QUICKMLP_NAMESPACE_END
