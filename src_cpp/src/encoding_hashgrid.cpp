#include <qmlp/encoding_hashgrid.h>

#include <tinyformat.h>
#include <magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <qmlp/kernels/encoding_hashgrid_config.cuh>
#include <qmlp/kernels/loops.cuh>
#include <qmlp/qmlp.h>

QUICKMLP_KERNEL_NAMESPACE_BEGIN

NLOHMANN_JSON_SERIALIZE_ENUM(LayerCombinationMode, {
    {LayerCombinationMode::CONCAT, "concat"},
    {LayerCombinationMode::ADD, "add"}
    })

QUICKMLP_KERNEL_NAMESPACE_END

QUICKMLP_NAMESPACE_BEGIN

static std::vector<float> vectorOrUnit(const std::vector<float>& vx, int d, float v)
{
    if (vx.empty())
    {
        return std::vector<float>(d, v);
    }
    else
    {
        CHECK_ERROR(vx.size() == d, "If a bounding box vector is specified, its dimension must match the configured dimension. Expected ", d, ", actual ", vx.size());
        return vx;
    }
}

static std::vector<float> invert(const std::vector<float>& vx)
{
    std::vector<float> o(vx.size());
    for (size_t i = 0; i < vx.size(); ++i) o[i] = 1.f / vx[i];
    return o;
}

EncodingHashGrid::EncodingHashGrid(int start_channel, int dimension, int num_levels, int num_features_per_level,
        int log2_hashmap_size, int min_resolution, int max_resolution, LayerCombinationMode combination_mode,
        const std::vector<float>& bounding_box_min, const std::vector<float>& bounding_box_size):
    startChannel_(start_channel),
    dimension_(dimension),
    numLevels_(num_levels),
    numFeaturesPerLevel_(num_features_per_level),
    log2HashmapSize_(log2_hashmap_size),
    hashmapSize_(log2_hashmap_size>0 ? 1 << log2_hashmap_size : -1),
    minResolution_(min_resolution),
    maxResolution_(max_resolution),
    combinationMode_(combination_mode),
    boundingBoxMin_(vectorOrUnit(bounding_box_min, dimension, -0.5f)),
    boundingBoxSize_(vectorOrUnit(bounding_box_size, dimension, 1.0f)),
    boundingBoxInvSize_(invert(boundingBoxSize_))
{
    CHECK_ERROR(dimension >= 1 && dimension <= 6,
        "Expected 'dimension' to be in [1,6], but got ", dimension);
    CHECK_ERROR(num_levels > 0, "num_levels must be positive but is ", num_levels);
    CHECK_ERROR(num_features_per_level > 0, "num_features_per_level must be positive but is ", num_features_per_level);
    CHECK_ERROR(min_resolution > 0, "min_resolution must be positive but is ", min_resolution);
    CHECK_ERROR(max_resolution >= min_resolution, "max_resolution must be larger than min_resolution");

    //calculate layer configs
    numParameters_ = 0;

    double levelScale = 1;
    if (num_levels > 1) {
        levelScale = std::exp((std::log(max_resolution) - std::log(min_resolution)) / (num_levels - 1));
    }

    for (int l=0; l<num_levels; ++l)
    {
        int resolution = static_cast<int>(std::floor(min_resolution * (std::pow(levelScale, l))));
        size_t requiredCells = ipow<size_t>(resolution, dimension);
        if (hashmapSize_ < 0 || requiredCells <= hashmapSize_)
        {
            //dense grid
            layers_.push_back({
                numParameters_, resolution, 0, false
            });
            numParameters_ += static_cast<int>(requiredCells) * num_features_per_level;
            QuickMLP::Instance().getLogger()->debug("HashGrid layer {} is dense. Resolution={}",
                l, resolution);
        }
        else
        {
            //sparse grid
            layers_.push_back({
                numParameters_, resolution, hashmapSize_, true
                });
            numParameters_ += hashmapSize_ * num_features_per_level;
            QuickMLP::Instance().getLogger()->debug("HashGrid layer {} is sparse and will be hashed. Resolution={}, hashmap size={}",
                l, resolution, hashmapSize_);
        }
    }
}

EncodingHashGrid::EncodingHashGrid(const nlohmann::json& cfg)
    : EncodingHashGrid(
        cfg.at("start_in").get<int>(),
        cfg.at("n_in").get<int>(),
        cfg.at("n_levels").get<int>(),
        cfg.at("n_features_per_level").get<int>(),
        cfg.at("log2_hashmap_size").get<int>(),
        cfg.at("min_resolution").get<int>(),
        cfg.at("max_resolution").get<int>(),
        cfg.at("combination_mode").get<LayerCombinationMode>(),
        cfg.value("bounding_box_min", std::vector<float>()),
        cfg.value("bounding_box_size", std::vector<float>())
    )
{}

nlohmann::json EncodingHashGrid::toJson() const
{
    return nlohmann::json{
        {"id", id()},
        {"start_in", startChannel_},
        {"n_in", dimension_},
        {"n_levels", numLevels_},
        {"n_features_per_level", numFeaturesPerLevel_},
        {"log2_hashmap_size", log2HashmapSize_},
        {"min_resolution", minResolution_},
        {"max_resolution", maxResolution_},
        {"combination_mode", combinationMode_},
        {"bounding_box_min", boundingBoxMin_},
        {"bounding_box_size", boundingBoxSize_}
    };
}

std::string EncodingHashGrid::ID()
{
    return "HashGrid";
}

std::string EncodingHashGrid::id() const
{
    return ID();
}

int EncodingHashGrid::maxInputChannel() const
{
    return startChannel_ + dimension_ - 1;
}

int EncodingHashGrid::numOutputChannels() const
{
    switch (combinationMode_)
    {
    case LayerCombinationMode::CONCAT: return numLevels_ * numFeaturesPerLevel_;
    case LayerCombinationMode::ADD: return numFeaturesPerLevel_;
    default: throw std::runtime_error("Unknown enum");
    }
}

std::string EncodingHashGrid::qualifiedName() const
{
    return tinyformat::format("%s::EncodingHashGrid<%d,%d,%d,%d,%s::LayerCombinationMode::%s>",
        CKL_STR(QUICKMLP_KERNEL_NAMESPACE),
        startChannel_, dimension_, numLevels_, numFeaturesPerLevel_,
        CKL_STR(QUICKMLP_KERNEL_NAMESPACE), magic_enum::enum_name(combinationMode_));
}

void EncodingHashGrid::fillCode(std::stringstream& code) const
{
    code << ckl::KernelLoader::MainFile("qmlp/kernels/encoding_hashgrid.cuh");
}

bool EncodingHashGrid::hasParameters() const
{
    return true;
}

std::string EncodingHashGrid::parameterName() const
{
    return tinyformat::format("%s::HashGridConfig<%d,%d>",
        CKL_STR(QUICKMLP_KERNEL_NAMESPACE), ndim(), numLevels());
}

Tensor::Precision EncodingHashGrid::parameterPrecision(Tensor::Usage usage) const
{
    return Tensor::FLOAT;
}

int EncodingHashGrid::parameterCount() const
{
    return numParameters_;
}

void EncodingHashGrid::setParameter(const Tensor& tensor, Tensor::Usage usage)
{
    if (tensor.defined())
    {
        CHECK_DTYPE(tensor, parameterPrecision(usage));
        CHECK_DIM(tensor, 1);
        CHECK_SIZE(tensor, 0, parameterCount());
    }
    if (usage == Tensor::INFERENCE)
        parametersForward_ = tensor;
    else
        parametersGradients_ = tensor;
}

int EncodingHashGrid::fillParameterMemory(char* dst, int dstSize)
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

    float* parametersForward = parametersForward_.defined() ? parametersForward_.dataPtr<float>() : nullptr;
    float* parametersBackward = parametersGradients_.defined() ? parametersGradients_.dataPtr<float>() : nullptr;
    addWithPadding(&parametersForward, sizeof(float*));
    addWithPadding(&parametersBackward, sizeof(float*));

    for (size_t i = 0; i < layers_.size(); ++i)
        addWithPadding(&layers_[i], sizeof(kernel::HashGridLayerConfig), i == 0 ? PADDING : 1);

    addWithPadding(boundingBoxMin_.data(), dimension_ * sizeof(float));
    addWithPadding(boundingBoxInvSize_.data(), dimension_ * sizeof(float));

    return index;
}

void EncodingHashGrid::fillParameterConstant(const std::string& constantName, const ckl::KernelFunction& function,
                                             CUstream stream)
{
    static std::vector<char> MEMORY(1024 * 1024);

    int index = fillParameterMemory(MEMORY.data(), MEMORY.size());

    CUdeviceptr dst = function.constant(constantName);
    CKL_SAFE_CALL(cuMemcpyHtoDAsync(dst, MEMORY.data(), index, stream));
}

void EncodingHashGrid::zeroGradients()
{
    if (parametersGradients_.defined())
        parametersGradients_.zero_();
}

QUICKMLP_NAMESPACE_END
