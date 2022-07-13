#pragma once

#include <cuda_fp16.h>
#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

#include "common.cuh"
#include "encoding_hashgrid_config.cuh"

QUICKMLP_KERNEL_NAMESPACE_BEGIN

template<int NumDimensions> using fposition_t = StaticArray<float, NumDimensions>;
template<int NumDimensions> using iposition_t = StaticArray<int, NumDimensions>;
template<int NumFeaturesPerLayer> using feature_t = StaticArray<float, NumFeaturesPerLayer>;

constexpr const int MaxHashDimensions = 6;
template<int N> struct HashValue;
template<> struct HashValue<0> { static constexpr int hash = 1; };
template<> struct HashValue<1> { static constexpr int hash = 2654435761; };
template<> struct HashValue<2> { static constexpr int hash = 805459861; };
template<> struct HashValue<3> { static constexpr int hash = 73856093; };
template<> struct HashValue<4> { static constexpr int hash = 19349663; };
template<> struct HashValue<5> { static constexpr int hash = 834927; };

/**
 * \brief Computes the index into the latent grid, both for dense and hash grids
 * \tparam NumDimensions the dimension of the latent grid
 * \tparam CurrentN internal use for the recursion
 */
template<int NumDimensions, int CurrentN=NumDimensions-1>
struct Indexing
{
    typedef StaticArray<float, NumDimensions> position_t;

    __forceinline__ __host__ __device__ static int indexDense(const iposition_t<NumDimensions>& position, int resolution)
    {
        //last coordinate is fastest
        return position[CurrentN] + indexDense<CurrentN - 1>(position, resolution) * resolution;
    }

    __forceinline__ __host__ __device__ static int indexHash(const iposition_t<NumDimensions>& position)
    {
        static_assert(NumDimensions <= MaxHashDimensions, "NumDimensions exceeds the supported dimensions for hashing (no more hash seeds defined)");
        return (position[CurrentN] * HashValue<CurrentN>::hash) ^ indexHash<CurrentN - 1>(position);
    }
};
template<int NumDimensions>
struct Indexing<NumDimensions, 0>
{
    __forceinline__ __host__ __device__ static int indexDense(const iposition_t<NumDimensions>& position, int resolution)
    {
        return position[0];
    }

    __forceinline__ __host__ __device__ static int indexHash(const iposition_t<NumDimensions >& position)
    {
        return (position[0] * HashValue<0>::hash);
    }
};

/**
 * \brief Fetches the feature at a given position in the grid (dense or hashed)
 * \tparam NumDimensions the dimension of the grid
 * \tparam NumFeaturesPerLayer the number of features per layer
 */
template<int NumDimensions, int NumFeaturesPerLayer>
struct ValueFetcher
{
    const HashGridLayerConfig& cfg;
    const float* __restrict__ parameters;
    feature_t<NumFeaturesPerLayer>& feature;

    __forceinline__ __host__ __device__ ValueFetcher(const HashGridLayerConfig& cfg, const float* __restrict__ parameters,
        feature_t<NumFeaturesPerLayer>& feature)
            : cfg(cfg), parameters(parameters), feature(feature)
    {}

    __forceinline__ __host__ __device__ void fetchAndAddValue(const iposition_t<NumDimensions>& ipos, float alpha)
    {
        int idx;
        if (cfg.hashed)
        {
            idx = Indexing<NumDimensions>::indexHash(ipos) % cfg.hashGridSize;
        }
        else
        {
            idx = Indexing<NumDimensions>::indexDense(ipos, cfg.resolution);
        }
        idx *= NumFeaturesPerLayer;

#pragma unroll
        for (int i=0; i<NumFeaturesPerLayer; ++i)
        {
            feature[i] += alpha * parameters[cfg.memoryOffset + idx + i];
        }
    }
};

/**
 * \brief Linear interpolation in the N-dimensional hypercube
 * \tparam NumDimensions the dimension of the hypercube
 * \tparam Fetcher the type of the value fetcher. See \ref ValueFetcher for the implementation
 * \tparam CurrentN internal use for the recursion
 */
template<int NumDimensions, typename Fetcher, int CurrentN=NumDimensions-1>
struct HypercubeInterpolator
{
    __host__ __device__ static void interpolate(const Fetcher& fetcher,
        const fposition_t<NumDimensions>& fpos, const iposition_t<NumDimensions>& iposL, const iposition_t<NumDimensions>& iposH, float alpha=1)
    {
        //interpolation of N-D is a linear interpolation in (N-1)-D
        HypercubeInterpolator<NumDimensions, Fetcher, CurrentN - 1>::interpolate(
            fetcher, fpos, iposL, iposH, alpha * (1 - fpos[CurrentN]));
        HypercubeInterpolator<NumDimensions, Fetcher, CurrentN - 1>::interpolate(
            fetcher, fpos, iposL.replace(CurrentN, iposH[CurrentN]), iposH, alpha * fpos[CurrentN]);
    }
};
template<int NumDimensions, typename Fetcher>
struct HypercubeInterpolator<NumDimensions, Fetcher, 0>
{
    __host__ __device__ static void interpolate(const Fetcher& fetcher,
        const fposition_t<NumDimensions>& fpos, const iposition_t<NumDimensions>& iposL, const iposition_t<NumDimensions>& iposH, float alpha = 1)
    {
        //linear interpolation in 1D
        fetcher.fetchAndAddValue(iposL, alpha * (1 - fpos[0]));
        fetcher.fetchAndAddValue(iposL.replace(0, iposH[0]), alpha * fpos[0]);
    }
};

/**
 * \brief CUDA implementation of a multi-resolution latent grid (dense+hashed)
 * \tparam StartChannel the start channel in the input array
 * \tparam NumDimensions the number of dimensions, 1 to 6
 * \tparam NumLayers the number of layers in the multi-resolution grid
 * \tparam NumFeaturesPerLayer the number of features per layer
 * \tparam CombinationMode how the features sampled from the layers should be combined
 */
template<int StartChannel, int NumDimensions, int NumLayers, int NumFeaturesPerLayer, LayerCombinationMode CombinationMode>
struct EncodingHashGrid
{
    typedef HashGridConfig<NumDimensions, NumLayers> param_t;

    /**
     * \brief Computes the forward code for a single layer
     * \tparam Add true iff the features should be added to the output, false if it should be assigned
     * \param position the position in the unit (hyper-)cube
     * \param cfg the layer configuration
     * \param parameters the parameter tensor
     * \param output the output (shared memory)
     */
    template<bool Add>
    static __device__ void forwardLayer(
        const fposition_t& position, const HashGridLayerConfig& cfg, 
        const float* __restrict__ parameters, half* output)
    {
        //get corner positions and interpolation values
        fposition_t fpos;
        iposition_t iposL, iposH;
        const int rm1 = cfg.resolution - 1;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            float p = position[i] * (cfg.resolution - 1);
            iposL[i] = max(0, min(rm1, static_cast<int>(p)));
            iposH[i] = max(0, min(rm1, static_cast<int>(p)+1));
            fpos[i] = p - static_cast<float>(iposL[i]);
        }

        //interpolate feature
        feature_t feature{ zero_initialization_tag() };
        ValueFetcher<NumDimensions, NumFeaturesPerLayer> fetcher(cfg, parameters, feature);

        //write to output
        if constexpr (Add)
        {
#pragma unroll
            for (int i = 0; i < NumFeaturesPerLayer; ++i)
                output[i] = __hadd(output[i], feature[i]);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NumFeaturesPerLayer; ++i)
                output[i] = __float2half(feature[i]);
        }
    }

    template<typename I>
    static __device__ void forward(const I input, half* output, const param_t& params)
    {
        //fetch position and transform to unit cube
        fposition_t position;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            position[i] = (input[i + StartChannel] - params.boundingBoxMin[i]) * params.boundingBoxInvSize[i];
        }

        //loop over the layers
        if constexpr (CombinationMode == LayerCombinationMode::CONCAT)
        {
            for (int l = 0; l < NumLayers; ++l)
                forwardLayer<false>(position, params.layers[l], params.parametersForward, 
                    output + (l * NumFeaturesPerLayer));
        }
        else if constexpr (CombinationMode == LayerCombinationMode::ADD)
        {
            forwardLayer<false>(position, params.layers[0], params.parametersForward, output);
            for (int l = 1; l < NumLayers; ++l)
                forwardLayer<true>(position, params.layers[l], params.parametersForward, output);
        }
    }

    template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O>
    static __device__ void adjoint(const I& input, const half* adjOutput, O& adjInput, const param_t& params)
    {
        //TODO: implement
    }
};


QUICKMLP_KERNEL_NAMESPACE_END
