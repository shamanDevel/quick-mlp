#pragma once

#include "common.h"
#include "iencoding.h"
#include <nlohmann/json.hpp>

#include "kernels/encoding_hashgrid_config.cuh"

QUICKMLP_NAMESPACE_BEGIN

typedef kernel::LayerCombinationMode LayerCombinationMode;

/**
 * Multi-Resolution hash grid 1D-6D.
 * For each layer:
 *  If enough memory is available, use a dense grid, else use hashing
 * The per-layer features are either concatenated or added together.
 *
 * The input coordinates are first transformed to the unit (hyper-) cube
 * before hashing. See boundingBoxMin and boundingBoxMax
 */
class EncodingHashGrid : public IEncoding
{

private:
    const int startChannel_; //the start channel into the input
    const int dimension_; //the dimension of the grid, 1D to 6D
    const int numLevels_; //the number of layers
    const int numFeaturesPerLevel_; //the number of features per layer
    const int log2HashmapSize_; //the log2 of the cells per layers
    const int hashmapSize_; //the number of cells per layer, derived from log2HashmapSize_
    const int minResolution_; //the resolution of the coarsest layer
    const int maxResolution_; //the resolution of the finest layer
    const LayerCombinationMode combinationMode_;
    const std::vector<float> boundingBoxMin_; //min point of the bounding box
    const std::vector<float> boundingBoxSize_;
    const std::vector<float> boundingBoxInvSize_;

    int numParameters_;
    std::vector<QUICKMLP_KERNEL_NAMESPACE::HashGridLayerConfig> layers_;
    Tensor parametersForward_;
    Tensor parametersGradients_;

public:
    EncodingHashGrid(int start_channel, int dimension, int num_levels, int num_features_per_level,
        int log2_hashmap_size, int min_resolution, int max_resolution, LayerCombinationMode combination_mode,
        const std::vector<float>& bounding_box_min, const std::vector<float>& bounding_box_size);

    EncodingHashGrid(const nlohmann::json& cfg);

    [[nodiscard]] nlohmann::json toJson() const override;
    static std::string ID();
    [[nodiscard]] std::string id() const override;

    [[nodiscard]] int maxInputChannel() const override;
    [[nodiscard]] int numOutputChannels() const override;
    [[nodiscard]] std::string qualifiedName() const override;
    void fillCode(std::stringstream& code) const override;
    [[nodiscard]] bool hasParameters() const override;
    [[nodiscard]] std::string parameterName() const override;
    [[nodiscard]] Tensor::Precision parameterPrecision(Tensor::Usage usage) const override;
    [[nodiscard]] size_t parameterCount() const override;
    void setParameter(const Tensor& tensor, Tensor::Usage usage) override;
    void fillParameterConstant(const std::string& constantName, const ckl::KernelFunction& function,
        CUstream stream) override;
    void zeroGradients() override;

    [[nodiscard]] int startChannel() const
    {
        return startChannel_;
    }

    [[nodiscard]] int dimension() const
    {
        return dimension_;
    }

    [[nodiscard]] int numLevels() const
    {
        return numLevels_;
    }

    [[nodiscard]] int numFeaturesPerLevel() const
    {
        return numFeaturesPerLevel_;
    }

    [[nodiscard]] int log2HashmapSize() const
    {
        return log2HashmapSize_;
    }

    [[nodiscard]] int hashmapSize() const
    {
        return hashmapSize_;
    }

    [[nodiscard]] int minResolution() const
    {
        return minResolution_;
    }

    [[nodiscard]] int maxResolution() const
    {
        return maxResolution_;
    }

    [[nodiscard]] LayerCombinationMode combinationMode() const
    {
        return combinationMode_;
    }

    [[nodiscard]] std::vector<float> boundingBoxMin() const
    {
        return boundingBoxMin_;
    }

    [[nodiscard]] std::vector<float> boundingBoxSize() const
    {
        return boundingBoxSize_;
    }
};

QUICKMLP_NAMESPACE_END
