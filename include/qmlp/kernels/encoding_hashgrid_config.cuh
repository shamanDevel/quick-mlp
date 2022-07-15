#pragma once

#include "common.cuh"

QUICKMLP_KERNEL_NAMESPACE_BEGIN

/**
 * How are the features of the individual layers combined?
 */
enum class LayerCombinationMode
{
    /**
     * The features are concatenated
     * --> #output features = #per-layer features * num layers
     */
    CONCAT,
    /**
     * The features are added together
     * --> #output features = #per-layer features
     */
     ADD
};
struct alignas(32) HashGridLayerConfig
{
    int memoryOffset; //offset in floats from the start of the float array)
    int resolution; //the per-side resolution of this layer
    int hashGridSize; //the number of entries if hashed
    bool hashed; //true iff hashed
};
template<int NumDimensions, int NumLayers>
struct HashGridConfig
{
    alignas(32) HashGridLayerConfig layers[NumLayers];
    alignas(32) float* __restrict__ parametersForward;
    alignas(32) float* parametersBackward;
    alignas(32) StaticArray<float, NumDimensions> boundingBoxMin;
    alignas(32) StaticArray<float, NumDimensions> boundingBoxInvSize;
};


QUICKMLP_KERNEL_NAMESPACE_END
