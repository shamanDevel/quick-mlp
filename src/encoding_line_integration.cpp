#include <qmlp/encoding_line_integration.h>

#include <tinyformat.h>

#include "qmlp/kernels/loops.cuh"

QUICKMLP_KERNEL_NAMESPACE_BEGIN
    NLOHMANN_JSON_SERIALIZE_ENUM(LineIntegrationBlendingMode, {
    {LineIntegrationBlendingMode::ADDITIVE, "additive"},
    {LineIntegrationBlendingMode::AVERAGE, "average"}
    })

QUICKMLP_KERNEL_NAMESPACE_END

QUICKMLP_NAMESPACE_BEGIN

EncodingLineIntegration::EncodingLineIntegration(int start_channel, int dimension,
    std::shared_ptr<IVolumetricEncoding> volume, float stepsize, kernel::LineIntegrationBlendingMode blendingMode): startChannel_(start_channel)
    , dimension_(dimension)
    , volume_(std::move(volume))
    , stepsizeWorld_(stepsize)
    , blendingMode_(blendingMode)
{
    CHECK_ERROR(stepsize > 0, "Stepsize must be positive, but is ", stepsize);

    CHECK_ERROR(dimension_ == volume_->ndim(), "Dimension of the child volume doesn't match");
    CHECK_ERROR(volume_->hasParameters(), "Child volume needs trainable parameters");
}

static std::shared_ptr<IVolumetricEncoding> parseChild(nlohmann::json cfg, int ndim)
{
    cfg["start_in"] = 0;
    cfg["n_in"] = ndim;
    auto ptr = EncodingFactory::Instance().create(cfg);
    auto ptrCast = std::dynamic_pointer_cast<IVolumetricEncoding>(ptr);
    CHECK_ERROR(ptrCast != nullptr, "Child encoding for LineIntegration could not be created");
    return ptrCast;
}

EncodingLineIntegration::EncodingLineIntegration(const nlohmann::json& cfg)
    : EncodingLineIntegration(
        cfg.at("start_in").get<int>(),
        cfg.at("n_in").get<int>(),
        parseChild(cfg.at("volume"), cfg.at("n_in").get<int>()),
        cfg.at("stepsize").get<float>(),
        cfg.at("blending").get<kernel::LineIntegrationBlendingMode>()
    )
{}

nlohmann::json EncodingLineIntegration::toJson() const
{
    return nlohmann::json{
        {"id", id()},
        {"start_in", startChannel_},
        {"n_in", dimension_},
        {"volume", volume_->toJson()},
        {"stepsize", stepsizeWorld_},
        {"blending", blendingMode_},
    };
}

std::string EncodingLineIntegration::ID()
{
    return "LineIntegration";
}

std::string EncodingLineIntegration::id() const
{
    return ID();
}

int EncodingLineIntegration::maxInputChannel() const
{
    return startChannel_ + dimension_ - 1;
}

int EncodingLineIntegration::numOutputChannels() const
{
    return volume_->numOutputChannels();
}

std::string EncodingLineIntegration::qualifiedName() const
{
    throw std::logic_error("Not implemented");
}

void EncodingLineIntegration::fillCode(std::stringstream& code) const
{
    code << ckl::KernelLoader::MainFile("qmlp/kernels/encoding_line_integration.cuh");
}

int EncodingLineIntegration::ndim() const
{
    return dimension_ * 2; //*2 for ray start + ray direction
}

IVolumetricEncoding::BoundingBoxVector_t EncodingLineIntegration::boundingBoxMin() const
{
    return volume_->boundingBoxMin();
}

IVolumetricEncoding::BoundingBoxVector_t EncodingLineIntegration::boundingBoxSize() const
{
    return volume_->boundingBoxSize();
}

IVolumetricEncoding::BoundingBoxVector_t EncodingLineIntegration::boundingBoxInvSize() const
{
    return volume_->boundingBoxInvSize();
}

bool EncodingLineIntegration::hasParameters() const
{
    return volume_->hasParameters();
}

std::string EncodingLineIntegration::parameterName() const
{
    const std::string childType = volume_->parameterName();
    return tinyformat::format("%s::LineIntegrationConfig<%d,%s>",
        CKL_STR(QUICKMLP_KERNEL_NAMESPACE), dimension_, childType);
}

Tensor::Precision EncodingLineIntegration::parameterPrecision(Tensor::Usage usage) const
{
    return volume_->parameterPrecision(usage);
}

int EncodingLineIntegration::parameterCount() const
{
    return volume_->parameterCount();
}

void EncodingLineIntegration::setParameter(const Tensor& tensor, Tensor::Usage usage)
{
    volume_->setParameter(tensor, usage);
}

int EncodingLineIntegration::fillParameterMemory(char* dst, int dstSize)
{
#define PADDING 8

    size_t index = 0;
    const auto addWithPadding = [&](const void* mem, size_t len, int padding = PADDING)
    {
        //add padding
        index = kernel::roundUpPower2(index, padding);
        if (len > 0) {
            assert(index + len < dstSize);
            memcpy(dst + index, mem, len);
            index += len;
        }
    };

    const auto& boundingBoxMin = this->boundingBoxMin();
    const auto& boundingBoxSize = this->boundingBoxSize();
    addWithPadding(boundingBoxMin.data(), dimension_ * sizeof(float));
    addWithPadding(boundingBoxSize.data(), dimension_ * sizeof(float));

    addWithPadding(&stepsizeWorld_, sizeof(float));

    //child
    index = kernel::roundUpPower2(index, PADDING);
    index += volume_->fillParameterMemory(dst + index, dstSize - index);

    return index;
}

void EncodingLineIntegration::fillParameterConstant(const std::string& constantName,
    const ckl::KernelFunction& function, CUstream stream)
{
    static std::vector<char> MEMORY(1024 * 1024);

    int index = fillParameterMemory(MEMORY.data(), MEMORY.size());

    CUdeviceptr dst = function.constant(constantName);
    CKL_SAFE_CALL(cuMemcpyHtoDAsync(dst, MEMORY.data(), index, stream));
}

void EncodingLineIntegration::zeroGradients()
{
    volume_->zeroGradients();
}

QUICKMLP_NAMESPACE_END
