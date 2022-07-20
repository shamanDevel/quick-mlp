#include <qmlp/fused_network.h>

#include <tinyformat.h>
#include <qmlp/qmlp.h>
#include <unordered_set>
#include <cmath>
#include <qmlp/kernels/tensor.cuh>

QUICKMLP_NAMESPACE_BEGIN

static int fetchSharedMemory()
{
    cudaDeviceProp props;
    CKL_SAFE_CALL(cudaGetDeviceProperties(&props, 0));
    if (props.warpSize != FusedNetwork::WARP_SIZE)
    {
        throw configuration_error("The warp size has changed, it is no longer %d but %d. This invalidates all algorithms!",
            FusedNetwork::WARP_SIZE, props.warpSize);
    }
    std::cout << "Available shared memory on the current device in bytes: " <<
        props.sharedMemPerBlock << std::endl;
    return static_cast<int>(props.sharedMemPerBlock);
}
const int FusedNetwork::MAX_SHARED_MEMORY_BYTES = fetchSharedMemory();

FusedNetwork::FusedNetwork(const nlohmann::json& cfg, const std::filesystem::path& parent)
    : channelsIn_(cfg.at("num_inputs").get<int>())
    , channelsOut_(cfg.at("num_outputs").get<int>())
    , networkInputPadding_(0)
    , numParameters_(0)
    , perEntryForwardMemoryBytes_(0)
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

    //create info for training
    backwardInfo_.resize(layers_.size());
    int offsetAdjOut = 0;
    int offsetIn = -1;
    for (int i=0; i<layers_.size(); ++i)
    {
        backwardInfo_[i].offsetAdjOut = offsetAdjOut;
        backwardInfo_[i].offsetIn = offsetIn;
        offsetIn = offsetAdjOut;
        offsetAdjOut += layers_[i].channelsOut;

        CKL_SAFE_CALL(cudaStreamCreateWithFlags(&backwardInfo_[i].stream, cudaStreamNonBlocking));
        CKL_SAFE_CALL(cudaEventCreateWithFlags(&backwardInfo_[i].event, cudaEventDisableTiming));
    }
    CKL_SAFE_CALL(cudaEventCreateWithFlags(&adjointEvent_, cudaEventDisableTiming));
}

FusedNetwork::~FusedNetwork()
{
    for (size_t i = 0; i < backwardInfo_.size(); ++i)
    {
        CKL_SAFE_CALL_NO_THROW(cudaStreamDestroy(backwardInfo_[i].stream));
        CKL_SAFE_CALL_NO_THROW(cudaEventDestroy(backwardInfo_[i].event));
    }
    CKL_SAFE_CALL_NO_THROW(cudaEventDestroy(adjointEvent_));
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

void FusedNetwork::initializeInferenceParameters(std::default_random_engine& rng)
{
    if (!parametersInference_.defined())
        throw std::runtime_error("Inference parameters not defined, call setNetworkParameter(...) first");
    assert(parametersInference_.precision() == Tensor::HALF);

    //create host memory
    std::vector<half> dataHost(parametersInference_.numel());

    //fill layers
    for (int layer=0; layer<numLayers(); ++layer)
    {
        const auto& l = layers_[layer];
        //kaiman uniform
        float bound = 1.0f / std::sqrt(l.channelsIn);
        std::uniform_real_distribution<float> distr1(-bound, +bound);
        //weights
        Tensor weights = networkParameter(layer, false, dataHost.data(), Tensor::HALF);
        for (int32_t i = 0; i < weights.size(0); ++i)
            for (int32_t j = 0; j < weights.size(1); ++j)
                weights.dataPtr<half>()[weights.idx({ i, j })] = __float2half(distr1(rng));
        //bias
        Tensor bias = networkParameter(layer, true, dataHost.data(), Tensor::HALF);
        if (l.useBias) {
            for (int32_t i = 0; i < bias.size(0); ++i)
                bias.dataPtr<half>()[bias.idx({ i })] = __float2half(distr1(rng));
        }
    }

    //copy to GPU
    CKL_SAFE_CALL(cudaMemcpy(parametersInference_.rawPtr(), dataHost.data(),
        parametersInference_.numel() * sizeof(half), cudaMemcpyHostToDevice));
}

Tensor FusedNetwork::networkParameter(int layer, bool bias, void* rawPtr, Tensor::Precision precision)
{
    if (layer < 0 || layer >= layers_.size())
        throw std::runtime_error("Layer index out of bounds");
    const auto& l = layers_[layer];

    int8_t* data = static_cast<int8_t*>(rawPtr);
    int bytesPerEntry = Tensor::BytesPerEntry[precision];
    if (bias)
    {
        if (!l.useBias) return {};
        return Tensor(
            data + bytesPerEntry * l.biasStart,
            precision,
            { l.channelsOut }, { 1 }
        );
    }
    else
    {
        //row-major
        return Tensor(
            data + bytesPerEntry * l.weightsStart,
            precision,
            { l.channelsOut, l.channelsIn }, { l.channelsIn, 1 }
        );
    }
}

Tensor FusedNetwork::networkParameter(int layer, bool bias, Tensor::Usage usage)
{
    auto& p = usage == Tensor::INFERENCE ? parametersInference_ : parametersGradients_;
    return networkParameter(layer, bias, p.rawPtr(), p.precision());
}

std::string FusedNetwork::constantNameForEncoding(IEncoding_ptr e, int encodingIdx)
{
    return tinyformat::format("cEncoding%d", encodingIdx);
}

void FusedNetwork::fillEncodingsAndActivations(std::string& codeTemplate,
                                               std::vector<std::string>& encodingConstantNames)
{
    //encodings
    std::stringstream encodingIncludes;
    std::stringstream encodingConstants;
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        e->fillCode(encodingIncludes);
        encodingIncludes << "\n";
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            encodingConstants << "__constant__ " << e->parameterName() << " " << constantName << "\n";
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
            if (activationIDs.count(a->id()) == 0)
            {
                activationDefinitions << a->code() << "\n";
                activationIDs.insert(a->id());
            }
        }
    }
    replaceAll(codeTemplate, "$$DEFINE_ACTIVATIONS$$", activationDefinitions.str());

}

void FusedNetwork::compileInferenceKernel()
{
    if (inferenceKernel_.has_value()) return;

    auto kl = QuickMLP::Instance().kernelLoader();
    std::string codeTemplate = kl->findFile("qmlp/kernels/network_forward.cuh").value();

    //GENERATE CODE
    std::vector<std::string> encodingConstantNames;
    fillEncodingsAndActivations(codeTemplate, encodingConstantNames);
    //constants
    int maxChannels = 0;
    for (const auto& l : layers_)
        maxChannels = std::max({ maxChannels, l.channelsIn, l.channelsOut });
    int maxEncodingChannels = 0;
    for (const auto& e : encodings_)
        maxEncodingChannels += e->numOutputChannels();
    int channelsIn = layers_.begin()->channelsIn;
    int channelsOut = layers_.rbegin()->channelsOut;
    int inputPadStart = maxEncodingChannels;
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
        callEncodings << "auto encodingOutput" << encodingOffset << " = encodingOutput + " << encodingOffset << ";\n";
        callEncodings << e->qualifiedName() << "::forward(encodingInput, encodingOutput" <<
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
    for (size_t layerIdx=0; layerIdx < layers_.size(); ++layerIdx)
    {
        const auto& l = layers_[layerIdx];
        //type
        if (l.activations.size() > 1) throw std::runtime_error("Individual activations per channel are not implemented yet");
        callLayers << "qmlp::kernel::Layer<" << (l.channelsIn / 16) << ", " << (l.channelsOut / 16) <<
            ", " << maxChannels << ", " << (l.useBias ? "true" : "false") << ", " <<
            "activations::" << l.activations[0]->id() << ">";
        //parameters
        callLayers << "::template inference<false>(networkParameters+" << l.weightsStart << ", ";
        if (l.useBias)
            callLayers << "networkParameters+" << l.biasStart << ", ";
        else
            callLayers << "nullptr, ";
        callLayers << "intermediateResultsWarp, nullptr);\n";
        ////test
        //callLayers << "if (index==0) {printLayer(" << (layerIdx + 1) << ", index, intermediateResultsThread, " <<
        //    l.channelsOut << ");}\n";
    }
    replaceAll(codeTemplate, "$$CALL_NETWORK_LAYERS$$", callLayers.str());

    //COMPILE
    int compileFlags = QuickMLP::Instance().getCompileFlags();
    ckl::KernelFunction fun = kl->getKernel(
        "qmlp::kernel::NetworkKernelInferenceAndForward",
        codeTemplate,
        encodingConstantNames,
        compileFlags).value();

    //compute shared memory
    int bestBlockSize = fun.bestBlockSize();
    int blockSize = bestBlockSize;
    int sharedMemorySize = bestBlockSize * maxChannels * sizeof(half);
    if (sharedMemorySize > MAX_SHARED_MEMORY_BYTES)
    {
        blockSize = MAX_SHARED_MEMORY_BYTES / (maxChannels * sizeof(half));
        blockSize = (blockSize / 32) * 32; //round up to multiple of the warp size
        std::cout << "It would be possible to fit more threads into each block in terms of register usage, but the shared memory is not enough. " <<
            "Reducing the block size from " << bestBlockSize << " down to " << blockSize << std::endl;
        sharedMemorySize = blockSize * maxChannels * sizeof(half);
    }

    //set cached kernel
    CompiledKernel ck;
    ck.fun = fun;
    ck.blockSize = blockSize;
    ck.sharedMemorySize = sharedMemorySize;
    ck.encodingIdx2ConstantName.resize(encodings_.size());
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            ck.encodingIdx2ConstantName[encodingIdx] = constantName;
            ck.encodingPtr2ConstantName[e] = constantName;
        }
    }

    inferenceKernel_ = std::move(ck);
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

    //compile kernel
    compileInferenceKernel();

    //CONSTANTS
    //fill constants of the encodings
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = inferenceKernel_->encodingIdx2ConstantName[encodingIdx];
            e->fillParameterConstant(constantName, inferenceKernel_->fun, stream);
        }
    }

    //LAUNCH KERNEL
    int minGridSize = std::min(
        CKL_DIV_UP(numel, inferenceKernel_->blockSize), 
        inferenceKernel_->fun.minGridSize());
    std::cout << "Launch with a block size of " << inferenceKernel_->blockSize << " and a shared memory size of " <<
        inferenceKernel_->sharedMemorySize << std::endl;
    //launch
    auto inputAcc = input.accessor<kernel::Tensor2Read<float>>();
    auto outputAcc = output.accessor<kernel::Tensor2RW<float>>();
    const half* networkParams = parametersInference_.dataPtr<half>();
    inferenceKernel_->fun.call(
        minGridSize, inferenceKernel_->blockSize, inferenceKernel_->sharedMemorySize, stream,
        numel, inputAcc, outputAcc, networkParams, nullptr);
}

void FusedNetwork::compileForwardKernel()
{
    if (forwardKernel_.has_value()) return;
    int perEntryForwardMemoryHalf = 0;

    auto kl = QuickMLP::Instance().kernelLoader();
    std::string codeTemplate = kl->findFile("qmlp/kernels/network_forward.cuh").value();

    //GENERATE CODE
    std::vector<std::string> encodingConstantNames;
    fillEncodingsAndActivations(codeTemplate, encodingConstantNames);
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
        callEncodings << "auto encodingOutput" << encodingOffset << " = encodingOutput + " << encodingOffset << ";\n";
        callEncodings << e->qualifiedName() << "::forward(encodingInput, encodingOutput" <<
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
        callLayers << "::template inference<true>(networkParameters+" << l.weightsStart << ", ";
        if (l.useBias)
            callLayers << "networkParameters+" << l.biasStart << ", ";
        else
            callLayers << "nullptr, ";
        callLayers << "intermediateResultsWarp, ";
        //storage for the intermediate states
        //offset into forwardTmpMemory:
        // prefix sum of the previous layers (perEntryForwardMemoryHalf) * numel
        // + 32 * channelsOut * warpID
        callLayers << "forwardTmpMemory + (" << perEntryForwardMemoryHalf << "*numel"
            << " + 32*" << l.channelsOut << "*warpID) ";
        perEntryForwardMemoryHalf += l.channelsOut;
        //end function
        callLayers << ");\n";
    }
    replaceAll(codeTemplate, "$$CALL_NETWORK_LAYERS$$", callLayers.str());

    //COMPILE
    int compileFlags = QuickMLP::Instance().getCompileFlags();
    ckl::KernelFunction fun = kl->getKernel(
        "qmlp::kernel::NetworkKernelInferenceAndForward",
        codeTemplate,
        encodingConstantNames,
        compileFlags).value();

    //compute shared memory
    int bestBlockSize = fun.bestBlockSize();
    int blockSize = bestBlockSize;
    int sharedMemorySize = bestBlockSize * maxChannels * sizeof(half);
    if (sharedMemorySize > MAX_SHARED_MEMORY_BYTES)
    {
        blockSize = MAX_SHARED_MEMORY_BYTES / (maxChannels * sizeof(half));
        blockSize = (blockSize / 32) * 32; //round up to multiple of the warp size
        std::cout << "It would be possible to fit more threads into each block in terms of register usage, but the shared memory is not enough. " <<
            "Reducing the block size from " << bestBlockSize << " down to " << blockSize << std::endl;
        sharedMemorySize = blockSize * maxChannels * sizeof(half);
    }

    //set cached kernel
    CompiledKernel ck;
    ck.fun = fun;
    ck.blockSize = blockSize;
    ck.sharedMemorySize = sharedMemorySize;
    ck.encodingIdx2ConstantName.resize(encodings_.size());
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            ck.encodingIdx2ConstantName[encodingIdx] = constantName;
            ck.encodingPtr2ConstantName[e] = constantName;
        }
    }

    forwardKernel_ = std::move(ck);
    perEntryForwardMemoryBytes_ = perEntryForwardMemoryHalf * sizeof(half);
}

size_t FusedNetwork::forwardMemory(int numElements, AdjointModeFlags adjointFlags)
{
    //compile kernel
    compileForwardKernel();

    numElements = roundUp(numElements, WARP_SIZE);
    return perEntryForwardMemoryBytes_ * static_cast<size_t>(numElements);
}

size_t FusedNetwork::forwardMemory(const Tensor& input, AdjointModeFlags adjointFlags)
{
    CHECK_DIM(input, 2);
    CHECK_SIZE(input, 1, channelsIn());
    int numel = input.size(0);
    return forwardMemory(numel, adjointFlags);
}

void FusedNetwork::forward(const Tensor& input, Tensor& output, void* tmpMemory, CUstream stream)
{
    CHECK_DIM(input, 2);
    CHECK_DIM(output, 2);
    CHECK_SIZE(input, 1, channelsIn());
    CHECK_SIZE(output, 1, channelsOut());
    int numel = input.size(0);
    CHECK_SIZE(output, 0, numel);
    CHECK_DTYPE(input, precisionIn());
    CHECK_DTYPE(output, precisionOut());

    //compile kernel
    compileForwardKernel();

    //CONSTANTS
    //fill constants of the encodings
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = forwardKernel_->encodingIdx2ConstantName[encodingIdx];
            e->fillParameterConstant(constantName, forwardKernel_->fun, stream);
        }
    }

    //LAUNCH KERNEL
    int minGridSize = std::min(
        CKL_DIV_UP(numel, forwardKernel_->blockSize),
        forwardKernel_->fun.minGridSize());
    std::cout << "Launch with a block size of " << forwardKernel_->blockSize << " and a shared memory size of " <<
        forwardKernel_->sharedMemorySize << std::endl;
    //launch
    auto inputAcc = input.accessor<kernel::Tensor2Read<float>>();
    auto outputAcc = output.accessor<kernel::Tensor2RW<float>>();
    const half* networkParams = parametersInference_.dataPtr<half>();
    half* tmpMemoryHalf = static_cast<half*>(tmpMemory);
    forwardKernel_->fun.call(
        minGridSize, forwardKernel_->blockSize, forwardKernel_->sharedMemorySize, stream,
        numel, inputAcc, outputAcc, networkParams, tmpMemoryHalf);
}

void FusedNetwork::zeroGradients()
{
    if (parametersGradients_.defined())
        parametersGradients_.zero_();
    for (auto& e : encodings_)
        e->zeroGradients();
}

void FusedNetwork::compileBackwardKernel(int flags)
{
    if (backwardKernel_[flags].has_value()) return;
    const bool hasInputGradients = flags & GRADIENTS_INPUT;
    const bool hasNetworkGradients = flags & GRADIENTS_NETWORK_WEIGHTS;
    const bool hasEncodingGradients = flags & GRADIENTS_INPUT_ENCODINGS;
    if (!hasInputGradients && !hasNetworkGradients && !hasEncodingGradients)
        throw std::runtime_error("At least one derivative must be enabled");

    AdjointKernel ak;
    ak.layers.resize(layers_.size());

    auto kl = QuickMLP::Instance().kernelLoader();
    std::string codeTemplate = kl->findFile("qmlp/kernels/network_backward.cuh").value();

    //GENERATE CODE
    std::vector<std::string> encodingConstantNames;
    fillEncodingsAndActivations(codeTemplate, encodingConstantNames);
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
    replaceAll(codeTemplate, "$$HAS_INPUT_GRADIENTS$$", hasInputGradients?"true":"false");

    //call layers
    //TODO: prefetch weights in shared memory?
    std::stringstream callLayers;
    for (int i=layers_.size()-1; i>=0; --i) //Note: reverse order!
    {
        const auto& l = layers_[i];
        auto& ali = backwardInfo_[i];
        //type
        if (l.activations.size() > 1) throw std::runtime_error("Individual activations per channel are not implemented yet");
        callLayers << "qmlp::kernel::Layer<" << (l.channelsIn / 16) << ", " << (l.channelsOut / 16) <<
            ", " << maxChannels << ", " << (l.useBias ? "true" : "false") << ", " <<
            "activations::" << l.activations[0]->id() << ">";
        //parameters
        callLayers << "::template adjoint< " <<
            (hasNetworkGradients ? "true" : "false") << " > (\n" <<
            "    networkParameters+" << l.weightsStart << ", ";
        if (l.useBias)
            callLayers << "networkParameters+" << l.biasStart << ", ";
        else
            callLayers << "nullptr, ";
        callLayers << "adjIntermediateResultsWarp, \n" <<
            "    forwardTmpMemory + (" << ali.offsetAdjOut << "*numel" << " + 32*" << l.channelsOut << "*warpID), " <<
            "adjointTmpMemory + (" << ali.offsetAdjOut << "*numel" << " + 32*" << l.channelsOut << "*warpID) );\n";
        ////test
        //callLayers << "if (index == 0) { printLayer(" << i << ", index, adjIntermediateResultsThread, " << l.channelsIn << "); }\n";
    }
    replaceAll(codeTemplate, "$$CALL_NETWORK_LAYERS$$", callLayers.str());


    //call encodings
    std::stringstream callEncodings;
    int encodingOffset = 0;
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        callEncodings << e->qualifiedName() << "::template adjoint<" <<
            (hasInputGradients ? "true" : "false") << ", " <<
            (hasEncodingGradients ? "true" : "false") <<
            ">(encodingInput, adjEncodingOutput + " << encodingOffset <<
            ", adjEncodingInput";
        encodingOffset += e->numOutputChannels();
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            callEncodings << ", " << constantName;
        }
        callEncodings << ");\n";
    }
    replaceAll(codeTemplate, "$$CALL_ENCODINGS$$", callEncodings.str());

    //COMPILE
    int compileFlags = QuickMLP::Instance().getCompileFlags();
    ckl::KernelFunction fun = kl->getKernel(
        "qmlp::kernel::NetworkKernelBackward",
        codeTemplate,
        encodingConstantNames,
        compileFlags).value();

    //compute shared memory
    int bestBlockSize = fun.bestBlockSize();
    int blockSize = bestBlockSize;
    int sharedMemorySize = bestBlockSize * maxChannels * sizeof(half);
    if (sharedMemorySize > MAX_SHARED_MEMORY_BYTES)
    {
        blockSize = MAX_SHARED_MEMORY_BYTES / (maxChannels * sizeof(half));
        blockSize = (blockSize / 32) * 32; //round down to multiple of the warp size
        std::cout << "It would be possible to fit more threads into each block in terms of register usage, but the shared memory is not enough. " <<
            "Reducing the block size from " << bestBlockSize << " down to " << blockSize << std::endl;
        sharedMemorySize = blockSize * maxChannels * sizeof(half);
    }

    //set cached kernel
    ak.adjoint.fun = fun;
    ak.adjoint.blockSize = blockSize;
    ak.adjoint.sharedMemorySize = sharedMemorySize;
    ak.adjoint.encodingIdx2ConstantName.resize(encodings_.size());
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = constantNameForEncoding(e, encodingIdx);
            ak.adjoint.encodingIdx2ConstantName[encodingIdx] = constantName;
            ak.adjoint.encodingPtr2ConstantName[e] = constantName;
        }
    }
    ak.perEntryAdjointMemoryBytes = hasNetworkGradients
        ? perEntryForwardMemoryBytes_ * sizeof(half)
        : 0;

    //now compile the kernels for the weight updates
    if (hasNetworkGradients)
    {
        ak.layers.resize(layers_.size());
        std::string codeTemplate = kl->findFile("qmlp/kernels/network_weight_update_block.cuh").value();

        //GENERATE CODE
        std::vector<std::string> encodingConstantNames;
        fillEncodingsAndActivations(codeTemplate, encodingConstantNames);
        replaceAll(codeTemplate, "$$INPUT_PAD_START$$", std::to_string(inputPadStart));
        replaceAll(codeTemplate, "$$CHANNELS_IN$$", std::to_string(channelsIn));
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

        for (int layer=0; layer<layers_.size(); ++layer)
        {
            const auto& forwardSpecs = layers_[layer];
            const auto& adjSpecs = backwardInfo_[layer];
            auto& ck = ak.layers[layer];

            if (forwardSpecs.useBias)
                throw configuration_error("Optimizing the bias is not supported (not implemented yet)");

            std::string kernelName;
            int sharedMemoryBytesPerWarp_Input = 0;
            if (layer==0)
            {
                //first layer, re-do the input parametrization again
                kernelName = tinyformat::format(
                    "qmlp::kernel::WeightUpdateSingleBlockKernel<%s, qmlp::kernel::OHatTmpLoader<%d>, qmlp::kernel::InputLoader<%d> >",
                    Tensor::DatatypePerEntry[networkParameterPrecision(Tensor::GRADIENTS)],
                    forwardSpecs.channelsOut / 16, forwardSpecs.channelsIn / 16);

                sharedMemoryBytesPerWarp_Input = sizeof(half) * 32 *
                    (forwardSpecs.channelsIn + forwardSpecs.channelsOut);
            }
            else
            {
                if (forwardSpecs.activations.size() > 1) throw std::runtime_error("Individual activations per channel are not implemented yet");
                kernelName = tinyformat::format(
                    "qmlp::kernel::WeightUpdateSingleBlockKernel<%s, qmlp::kernel::OHatTmpLoader<%d>, qmlp::kernel::HiddenLoader<%d, qmlp::kernel::activations::%s> >",
                    Tensor::DatatypePerEntry[networkParameterPrecision(Tensor::GRADIENTS)],
                    forwardSpecs.channelsOut / 16, forwardSpecs.channelsIn / 16,
                    forwardSpecs.activations[0]->id());

                sharedMemoryBytesPerWarp_Input = sizeof(half) * 32 *
                    (forwardSpecs.channelsIn + forwardSpecs.channelsOut);
            }
            int sharedMemoryBytesPerWarp_Output = //TODO: bias
                Tensor::BytesPerEntry[networkParameterPrecision(Tensor::GRADIENTS)] *
                forwardSpecs.channelsIn * forwardSpecs.channelsOut;
            int sharedMemoryBytesPerWarp = std::max(
                sharedMemoryBytesPerWarp_Input,
                sharedMemoryBytesPerWarp_Output);

            //compile
            ckl::KernelFunction fun = kl->getKernel(
                kernelName,
                codeTemplate,
                encodingConstantNames,
                compileFlags).value();

            //compute shared memory
            int bestBlockSize = fun.bestBlockSize();
            int blockSize = bestBlockSize;
            int sharedMemorySize = bestBlockSize / 32 * sharedMemoryBytesPerWarp;
            if (sharedMemorySize > MAX_SHARED_MEMORY_BYTES)
            {
                int numWarps = MAX_SHARED_MEMORY_BYTES / sharedMemoryBytesPerWarp;
                blockSize = numWarps * 32;
                if (blockSize == 0)
                {
                    throw configuration_error("Layer %d too large for block-wise weight update!", layer);
                }
                std::cout << "It would be possible to fit more threads into each block in terms of register usage, but the shared memory is not enough. " <<
                    "Reducing the block size from " << bestBlockSize << " down to " << blockSize << std::endl;
                sharedMemorySize = numWarps * sharedMemoryBytesPerWarp;
            }

            //set cached kernel
            ck.fun = fun;
            ck.blockSize = blockSize;
            ck.sharedMemorySize = sharedMemorySize;
            if (layer == 0) {
                ck.encodingIdx2ConstantName.resize(encodings_.size());
                for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
                {
                    auto e = encodings_[encodingIdx];
                    if (e->hasParameters())
                    {
                        std::string constantName = constantNameForEncoding(e, encodingIdx);
                        ak.adjoint.encodingIdx2ConstantName[encodingIdx] = constantName;
                        ak.adjoint.encodingPtr2ConstantName[e] = constantName;
                    }
                }
            }
        }
    }

    backwardKernel_[flags] = std::move(ak);
}

size_t FusedNetwork::adjointMemory(int numElements, AdjointModeFlags adjointFlags)
{
    compileBackwardKernel(adjointFlags);
    numElements = roundUp(numElements, WARP_SIZE);
    return backwardKernel_[adjointFlags]->perEntryAdjointMemoryBytes * static_cast<size_t>(numElements);
}

size_t FusedNetwork::adjointMemory(const Tensor& input, AdjointModeFlags adjointFlags)
{
    CHECK_DIM(input, 2);
    CHECK_SIZE(input, 1, channelsIn());
    int numel = input.size(0);
    return adjointMemory(numel, adjointFlags);
}

void FusedNetwork::adjoint(const Tensor& input, const Tensor& adjOutput, AdjointModeFlags adjointFlags,
                           Tensor& adjInput, const void* tmpMemoryForward, void* tmpMemoryAdjoint, CUstream stream)
{
    CHECK_DIM(input, 2);
    CHECK_DIM(adjOutput, 2);
    CHECK_SIZE(input, 1, channelsIn());
    CHECK_SIZE(adjOutput, 1, channelsOut());
    int numel = input.size(0);
    CHECK_SIZE(adjOutput, 0, numel);
    CHECK_DTYPE(input, precisionIn());
    CHECK_DTYPE(adjOutput, precisionOut());

    const bool hasInputGradients = adjointFlags & GRADIENTS_INPUT;
    const bool hasNetworkGradients = adjointFlags & GRADIENTS_NETWORK_WEIGHTS;
    const bool hasEncodingGradients = adjointFlags & GRADIENTS_INPUT_ENCODINGS;

    if (hasInputGradients)
    {
        CHECK_ERROR(adjInput.defined(), "Input gradients requested, but adjInput is undefined");
        CHECK_DIM(adjInput, 2);
        CHECK_SIZE(adjInput, 1, channelsIn());
        CHECK_SIZE(adjInput, 0, numel);
        CHECK_DTYPE(adjInput, precisionIn());
    }

    //compile kernel
    compileBackwardKernel(adjointFlags);
    auto& ak = backwardKernel_[adjointFlags].value();

    //CONSTANTS
    //fill constants of the encodings
    for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
    {
        auto e = encodings_[encodingIdx];
        if (e->hasParameters())
        {
            std::string constantName = ak.adjoint.encodingIdx2ConstantName[encodingIdx];
            e->fillParameterConstant(constantName, ak.adjoint.fun, stream);
        }
    }

    //LAUNCH KERNEL for the main network
    int minGridSize = std::min(
        CKL_DIV_UP(numel, ak.adjoint.blockSize),
        ak.adjoint.fun.minGridSize());
    std::cout << "Launch with a block size of " << ak.adjoint.blockSize << " and a shared memory size of " <<
        ak.adjoint.sharedMemorySize << std::endl;
    //launch
    auto inputAcc = input.accessor<kernel::Tensor2Read<float>>();
    auto adjOutputAcc = adjOutput.accessor<kernel::Tensor2Read<float>>();
    kernel::Tensor2RW<float> adjInputAcc;
    if (hasInputGradients) adjInputAcc = adjInput.accessor<kernel::Tensor2RW<float>>();
    const half* networkParams = parametersInference_.dataPtr<half>();
    const half* forwardTmpMemory = static_cast<const half*>(tmpMemoryForward);
    const half* adjointTmpMemory = static_cast<const half*>(tmpMemoryAdjoint);
    ak.adjoint.fun.call(
        minGridSize, ak.adjoint.blockSize, ak.adjoint.sharedMemorySize, stream,
        numel, inputAcc, adjOutputAcc, adjInputAcc, networkParams, forwardTmpMemory, adjointTmpMemory);
    CKL_SAFE_CALL(cudaEventRecord(adjointEvent_, stream));

    //TEST
    CKL_SAFE_CALL(cudaDeviceSynchronize());

    //LAUNCH KERNELS FOR WEIGHT+BIAS UPDATE
    if (hasNetworkGradients)
    {
        for (int layer=0; layer<layers_.size(); ++layer)
        {
            const auto& ls = layers_[layer];
            const auto& ali = backwardInfo_[layer];
            auto& ck = ak.layers[layer];

            if (!ck.fun.defined()) continue; //not available

            //fill constants of the encodings (first layer only, if needed at all)
            if (!ck.encodingIdx2ConstantName.empty()) {
                for (int encodingIdx = 0; encodingIdx < encodings_.size(); ++encodingIdx)
                {
                    auto e = encodings_[encodingIdx];
                    if (e->hasParameters())
                    {
                        std::string constantName = ck.encodingIdx2ConstantName[encodingIdx];
                        e->fillParameterConstant(constantName, ak.adjoint.fun, stream);
                    }
                }
            }

            //assemble parameter pointers
            void* outAdjWeights = static_cast<char*>(parametersGradients_.rawPtr()) +
                parametersGradients_.bytesPerEntry() * ls.weightsStart;
            half* aIn = static_cast<half*>(tmpMemoryAdjoint) + (ali.offsetAdjOut * numel);

            //launch kernel
            int gridSize = 1; //one block only!
            CKL_SAFE_CALL(cudaStreamWaitEvent(ali.stream, adjointEvent_));
            std::cout << "Launch layer " << layer << " with a block size of " << ck.blockSize << " and a shared memory size of " <<
                ck.sharedMemorySize << std::endl;
            if (ali.offsetIn>=0)
            {
                const half* bIn = static_cast<const half*>(tmpMemoryForward) + (ali.offsetIn * numel);
                ck.fun.call(gridSize, ck.blockSize, ck.sharedMemorySize, ali.stream,
                    numel, outAdjWeights, aIn, bIn);
            }
            else
            {
                auto bIn = inputAcc;
                ck.fun.call(gridSize, ck.blockSize, ck.sharedMemorySize, ali.stream,
                    numel, outAdjWeights, aIn, bIn);
            }
            CKL_SAFE_CALL(cudaEventRecord(ali.event, ali.stream));
            CKL_SAFE_CALL(cudaStreamWaitEvent(stream, ali.event));

            //TEST
            CKL_SAFE_CALL(cudaDeviceSynchronize());
        }
    }
}

void FusedNetwork::clearCachedKernels()
{
    inferenceKernel_ = {};
    forwardKernel_ = {};
    for (int i = 0; i < ADJOINT_MODE_MAX_COMBINATIONS; ++i)
        backwardKernel_[i] = {};
}


QUICKMLP_NAMESPACE_END
