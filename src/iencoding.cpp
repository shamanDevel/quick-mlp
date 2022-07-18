#include <qmlp/iencoding.h>

#include <qmlp/encoding_identity.h>
#include <qmlp/encoding_hashgrid.h>

#include "qmlp/qmlp.h"
#include "qmlp/kernels/tensor.cuh"

QUICKMLP_NAMESPACE_BEGIN

void IEncoding::forward(const Tensor& input, Tensor& output, CUstream stream,
    const std::optional<const Tensor>& parametersForward)
{
    //Check shapes
    CHECK_DIM(input, 2);
    CHECK_DIM(output, 2);
    CHECK_DTYPE(input, Tensor::FLOAT);
    CHECK_DTYPE(output, Tensor::FLOAT);
    CHECK_SIZE(output, 0, input.size(0));
    CHECK_ERROR(input.size(1) > maxInputChannel());
    CHECK_ERROR(output.size(1) >= numOutputChannels());
    CHECK_ERROR(output.strides()[1] == 1, "The output stride along the channels must be 1");
    long long numel = input.size(0);

    static const std::string constantName = "cEncoding";
    if (!forwardKernel_.has_value()) {
        //generate code
        auto kl = QuickMLP::Instance().kernelLoader();
        std::string codeTemplate = kl->findFile("qmlp/kernels/encoding_kernels.cuh").value();

        //encoding parameters
        std::stringstream encodingIncludes;
        std::stringstream encodingConstants;
        std::vector<std::string> constantNames;
        fillCode(encodingIncludes);
        encodingIncludes << "\n";
        if (hasParameters())
        {
            encodingConstants << "__constant__ " << parameterName() << " " << constantName << ";\n";
            constantNames.push_back(constantName);
        }
        replaceAll(codeTemplate, "$$INCLUDES$$", encodingIncludes.str());
        replaceAll(codeTemplate, "$$ENCODING_CONSTANTS$$", encodingConstants.str());

        //call encodings
        std::stringstream callEncodings;
        callEncodings << qualifiedName() << "::forward(encodingInput, encodingOutput";
        if (hasParameters())
        {
            callEncodings << ", " << constantName;
        }
        callEncodings << ");\n";
        replaceAll(codeTemplate, "$$CALL_ENCODINGS_FORWARD$$", callEncodings.str());
        replaceAll(codeTemplate, "$$CALL_ENCODINGS_ADJOINT$$", "");

        //compile
        int compileFlags = QuickMLP::Instance().getCompileFlags();
        ckl::KernelFunction fun = kl->getKernel(
            "qmlp::kernel::EncodingForwardKernel",
            codeTemplate,
            constantNames,
            compileFlags).value();

        forwardKernel_ = fun;
    }
    auto& fun = forwardKernel_.value();

    //fill constants of the encodings
    if (hasParameters())
    {
        if (parametersForward.has_value())
        {
            //overwrite parameters
            setParameter(parametersForward.value(), Tensor::INFERENCE);
        }
        fillParameterConstant(constantName, fun, stream);
    }

    //launch kernel
    int minGridSize = std::min(
        static_cast<int>(CKL_DIV_UP(numel, fun.bestBlockSize())),
        fun.minGridSize());
    auto inputAcc = input.accessor<kernel::Tensor2Read<float>>();
    auto outputAcc = output.accessor<kernel::Tensor2RW<float>>();
    fun.call(
        minGridSize, fun.bestBlockSize(), 0, stream,
        numel, inputAcc, outputAcc);

    if (hasParameters() && parametersForward.has_value())
    {
        //clear parameter
        setParameter({}, Tensor::INFERENCE);
    }
}

void IEncoding::adjoint(const Tensor& input, const Tensor& adjOutput, Tensor& adjInput, CUstream stream,
    const std::optional<const Tensor>& parametersForward, const std::optional<const Tensor>& parametersGradients)
{
    //Check shapes
    CHECK_DIM(input, 2);
    CHECK_DIM(adjOutput, 2);
    CHECK_DIM(adjInput, 2);
    CHECK_DTYPE(input, Tensor::FLOAT);
    CHECK_DTYPE(adjOutput, Tensor::FLOAT);
    CHECK_DTYPE(adjInput, Tensor::FLOAT);
    CHECK_SIZE(adjOutput, 0, input.size(0));
    CHECK_SIZE(adjInput, 0, input.size(0));
    CHECK_ERROR(input.size(1) > maxInputChannel());
    CHECK_ERROR(adjOutput.size(1) >= numOutputChannels());
    CHECK_ERROR(adjInput.size(1) == input.size(1));
    long long numel = input.size(0);

    static const std::string constantName = "cEncoding";
    if (!adjointKernel_.has_value()) {
        //generate code
        auto kl = QuickMLP::Instance().kernelLoader();
        std::string codeTemplate = kl->findFile("qmlp/kernels/encoding_kernels.cuh").value();

        //encoding parameters
        std::stringstream encodingIncludes;
        std::stringstream encodingConstants;
        std::vector<std::string> constantNames;
        fillCode(encodingIncludes);
        encodingIncludes << "\n";
        if (hasParameters())
        {
            encodingConstants << "__constant__ " << parameterName() << " " << constantName << ";\n";
            constantNames.push_back(constantName);
        }
        replaceAll(codeTemplate, "$$INCLUDES$$", encodingIncludes.str());
        replaceAll(codeTemplate, "$$ENCODING_CONSTANTS$$", encodingConstants.str());

        //call encodings
        std::stringstream callEncodings;
        callEncodings << qualifiedName() << "::adjoint<true, true>(encodingInput, encodingAdjOutput, encodingAdjInput";
        if (hasParameters())
        {
            callEncodings << ", " << constantName;
        }
        callEncodings << ");\n";
        replaceAll(codeTemplate, "$$CALL_ENCODINGS_FORWARD$$", "");
        replaceAll(codeTemplate, "$$CALL_ENCODINGS_ADJOINT$$", callEncodings.str());

        //compile
        int compileFlags = QuickMLP::Instance().getCompileFlags();
        ckl::KernelFunction fun = kl->getKernel(
            "qmlp::kernel::EncodingAdjointKernel",
            codeTemplate,
            constantNames,
            compileFlags).value();

        adjointKernel_ = fun;
    }
    auto& fun = adjointKernel_.value();

    //fill constants of the encodings
    if (hasParameters())
    {
        //overwrite parameters
        if (parametersForward.has_value())
        {
            setParameter(parametersForward.value(), Tensor::INFERENCE);
        }
        if (parametersGradients.has_value())
        {
            setParameter(parametersGradients.value(), Tensor::GRADIENTS);
        }

        fillParameterConstant(constantName, fun, stream);
    }

    //launch kernel
    int minGridSize = std::min(
        static_cast<int>(CKL_DIV_UP(numel, fun.bestBlockSize())),
        fun.minGridSize());
    auto inputAcc = input.accessor<kernel::Tensor2Read<float>>();
    auto adjOutputAcc = adjOutput.accessor<kernel::Tensor2Read<float>>();
    auto adjInputAcc = adjInput.accessor<kernel::Tensor2RW<float>>();
    fun.call(
        minGridSize, fun.bestBlockSize(), 0, stream,
        numel, inputAcc, adjOutputAcc, adjInputAcc);

    //clear parameter
    if (hasParameters() && parametersForward.has_value())
        setParameter({}, Tensor::INFERENCE);
    if (hasParameters() && parametersGradients.has_value())
        setParameter({}, Tensor::GRADIENTS);
}

EncodingFactory::EncodingFactory()
{
    encodings_[EncodingIdentity::ID()] = [](const nlohmann::json& cfg)
    {
        return std::make_shared<EncodingIdentity>(cfg);
    };
    encodings_[EncodingHashGrid::ID()] = [](const nlohmann::json& cfg)
    {
        return std::make_shared<EncodingHashGrid>(cfg);
    };
    //more encodings here
}

EncodingFactory& EncodingFactory::Instance()
{
    static EncodingFactory INSTANCE;
    return INSTANCE;
}

IEncoding_ptr EncodingFactory::create(const nlohmann::json& cfg)
{
    const std::string id = cfg.at("id");
    auto it = encodings_.find(id);
    if (it == encodings_.end())
        throw configuration_error("No encoding with id '%s' found", id.c_str());
    return it->second(cfg);
}

QUICKMLP_NAMESPACE_END
