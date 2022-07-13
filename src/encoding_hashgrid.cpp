#include <qmlp/encoding_hashgrid.h>

#include <tinyformat.h>

QUICKMLP_NAMESPACE_BEGIN

NLOHMANN_JSON_SERIALIZE_ENUM(LayerCombinationMode, {
    {LayerCombinationMode::CONCAT, "concat"},
    {LayerCombinationMode::ADD, "add"}
})

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
    boundingBoxSize_(vectorOrUnit(bounding_box_size, dimension, 1.0f))
{
    CHECK_ERROR(dimension >= 1 && dimension <= 6,
        "Expected 'dimension' to be in [1,6], but got ", dimension);
    CHECK_ERROR(num_levels > 0, "num_levels must be positive but is ", num_levels);
    CHECK_ERROR(num_features_per_level > 0, "num_features_per_level must be positive but is ", num_features_per_level);
    CHECK_ERROR(min_resolution > 0, "min_resolution must be positive but is ", min_resolution);
    CHECK_ERROR(max_resolution >= min_resolution, "max_resolution must be larger than min_resolution");
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
{
}

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
    return "hashgrid";
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
    throw std::logic_error("Not implemented");
}

void EncodingHashGrid::fillCode(std::stringstream& code) const
{
    throw std::logic_error("Not implemented");
}

QUICKMLP_NAMESPACE_END
