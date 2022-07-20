#pragma once

#include <cuda_fp16.h>
#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

#include <qmlp/kernels/common.cuh>
#include <qmlp/kernels/encoding_hashgrid_config.cuh>

QUICKMLP_KERNEL_NAMESPACE_BEGIN

template<int NumDimensions> using fposition_t = StaticArray<float, NumDimensions>;
template<int NumDimensions> using iposition_t = StaticArray<unsigned int, NumDimensions>;
template<int NumFeaturesPerLayer> using feature_t = StaticArray<float, NumFeaturesPerLayer>;

constexpr const int MaxHashDimensions = 6;
template<int N> struct HashValue;
template<> struct HashValue<0> { static constexpr unsigned int hash = 1; };
template<> struct HashValue<1> { static constexpr unsigned int hash = 43857811; };
template<> struct HashValue<2> { static constexpr unsigned int hash = 83857661; };
template<> struct HashValue<3> { static constexpr unsigned int hash = 73856093; };
template<> struct HashValue<4> { static constexpr unsigned int hash = 19349663; };
template<> struct HashValue<5> { static constexpr unsigned int hash = 834927; };

/**
 * \brief Computes the index into the latent grid, both for dense and hash grids
 * \tparam NumDimensions the dimension of the latent grid
 * \tparam CurrentN internal use for the recursion
 */
template<int NumDimensions, int CurrentN=NumDimensions-1>
struct Indexing
{
    typedef StaticArray<float, NumDimensions> position_t;

    __forceinline__ __host__ __device__ static unsigned int indexDense(const iposition_t<NumDimensions>& position, unsigned int resolution)
    {
        //last coordinate is fastest
        return position[CurrentN] + Indexing<NumDimensions, CurrentN - 1>::indexDense(position, resolution) * resolution;
    }

    __forceinline__ __host__ __device__ static unsigned int indexHash(const iposition_t<NumDimensions>& position)
    {
        static_assert(NumDimensions <= MaxHashDimensions, "NumDimensions exceeds the supported dimensions for hashing (no more hash seeds defined)");
        return (position[CurrentN] * HashValue<CurrentN>::hash) ^ Indexing<NumDimensions, CurrentN - 1>::indexHash(position);
    }
};
template<int NumDimensions>
struct Indexing<NumDimensions, 0>
{
    __forceinline__ __host__ __device__ static unsigned int indexDense(const iposition_t<NumDimensions>& position, unsigned int resolution)
    {
        return position[0];
    }

    __forceinline__ __host__ __device__ static unsigned int indexHash(const iposition_t<NumDimensions >& position)
    {
        return (position[0] * HashValue<0>::hash);
    }
};

template<int NumDimensions, int NumFeaturesPerLayer>
struct ValueFetcherBase
{
    const HashGridLayerConfig& cfg;

    __forceinline__ __device__ ValueFetcherBase(const HashGridLayerConfig& cfg)
        : cfg(cfg)
    {}

    __forceinline__ __device__ unsigned int index(const iposition_t<NumDimensions>& ipos)
    {
        unsigned int idx;
        if (cfg.hashed)
        {
            idx = Indexing<NumDimensions>::indexHash(ipos) % cfg.hashGridSize;
        }
        else
        {
            idx = Indexing<NumDimensions>::indexDense(ipos, cfg.resolution);
        }
        idx *= NumFeaturesPerLayer;
        return idx;
    }
};

/**
 * \brief Fetches the feature at a given position in the grid (dense or hashed)
 * \tparam NumDimensions the dimension of the grid
 * \tparam NumFeaturesPerLayer the number of features per layer
 */
template<int NumDimensions, int NumFeaturesPerLayer>
struct ValueFetcherForward : ValueFetcherBase<NumDimensions, NumFeaturesPerLayer>
{
    typedef ValueFetcherBase<NumDimensions, NumFeaturesPerLayer> Base;

    const float* __restrict__ parameters;
    feature_t<NumFeaturesPerLayer>& feature;

    __forceinline__ __device__ ValueFetcherForward(const HashGridLayerConfig& cfg, const float* __restrict__ parameters,
        feature_t<NumFeaturesPerLayer>& feature)
            : Base(cfg), parameters(parameters), feature(feature)
    {}

    __forceinline__ __device__ void fetchAndAddValue(const iposition_t<NumDimensions>& ipos, float alpha)
    {
        unsigned int idx = this->index(ipos);

#pragma unroll
        for (int i=0; i<NumFeaturesPerLayer; ++i)
        {
            feature[i] += alpha * __ldg(parameters + (this->cfg.memoryOffset + idx + i));
        }
    }
};

template<int NumDimensions, int NumFeaturesPerLayer, bool EvaluateParameterGradients>
struct ValueFetcherAdjoint : ValueFetcherBase<NumDimensions, NumFeaturesPerLayer>
{
    typedef ValueFetcherBase<NumDimensions, NumFeaturesPerLayer> Base;

    float* adjParameters;
    const feature_t<NumFeaturesPerLayer>& adjFeatures;

    __forceinline__ __device__ ValueFetcherAdjoint(const HashGridLayerConfig& cfg, float* adjParameters,
        const feature_t<NumFeaturesPerLayer>& adjFeatures)
        : Base(cfg), adjParameters(adjParameters), adjFeatures(adjFeatures)
    {}

    __forceinline__ __device__ void fetchAndAddValue(const iposition_t<NumDimensions>& ipos, float alpha)
    {
        if constexpr (EvaluateParameterGradients) {
            unsigned int idx = this->index(ipos);
#pragma unroll
            for (int i = 0; i < NumFeaturesPerLayer; ++i)
            {
                ::atomicAdd(adjParameters + (this->cfg.memoryOffset + idx + i), alpha * adjFeatures[i]);
            }
        }
    }
};

template<int NumDimensions, int NumFeaturesPerLayer>
struct ValueFetcherDot : ValueFetcherBase<NumDimensions, NumFeaturesPerLayer>
{
    typedef ValueFetcherBase<NumDimensions, NumFeaturesPerLayer> Base;

    const float* __restrict__ parameters;

    __forceinline__ __device__ ValueFetcherDot(const HashGridLayerConfig& cfg, const float* __restrict__ parameters)
        : Base(cfg), parameters(parameters)
    {}

    __forceinline__ __device__ float fetchAndDot(const iposition_t<NumDimensions>& ipos, const feature_t<NumFeaturesPerLayer>& rhs)
    {
        unsigned int idx = this->index(ipos);

        float o = 0;
#pragma unroll
        for (int i = 0; i < NumFeaturesPerLayer; ++i)
        {
            o += rhs[i] * __ldg(parameters + (this->cfg.memoryOffset + idx + i));
        }
        return o;
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
    __host__ __device__ static void interpolate(Fetcher& fetcher,
        const fposition_t<NumDimensions>& fpos, const iposition_t<NumDimensions>& iposL, const iposition_t<NumDimensions>& iposH, float alpha=1)
    {
        //interpolation of N-D is a linear interpolation in (N-1)-D
        HypercubeInterpolator<NumDimensions, Fetcher, CurrentN - 1>::interpolate(
            fetcher, fpos, iposL, iposH, alpha * (1 - fpos[CurrentN]));
        HypercubeInterpolator<NumDimensions, Fetcher, CurrentN - 1>::interpolate(
            fetcher, fpos, iposL.replace(CurrentN, iposH[CurrentN]), iposH, alpha * fpos[CurrentN]);
    }

    template<int NumFeaturesPerLayer>
    __host__ __device__ static void adjoint(Fetcher& fetcher,
        const fposition_t<NumDimensions>& fpos, const iposition_t<NumDimensions>& iposL, const iposition_t<NumDimensions>& iposH, 
        fposition_t<NumDimensions>& adjPosition, const feature_t<NumFeaturesPerLayer>& adjFeature,
        const fposition_t<NumDimensions>& alphas = fposition_t<NumDimensions>(1.f),
        const fposition_t<NumDimensions>& signs = fposition_t<NumDimensions>(1.f))
    {
        //interpolation of N-D is a linear interpolation in (N-1)-D
        HypercubeInterpolator<NumDimensions, Fetcher, CurrentN - 1>::template adjoint<NumFeaturesPerLayer>(
            fetcher, fpos, iposL, iposH, adjPosition, adjFeature,
            alphas.replace(CurrentN, 1 - fpos[CurrentN]),
            signs.replace(CurrentN, -1.f));
        HypercubeInterpolator<NumDimensions, Fetcher, CurrentN - 1>::template adjoint<NumFeaturesPerLayer>(
            fetcher, fpos, iposL.replace(CurrentN, iposH[CurrentN]), iposH, adjPosition, adjFeature,
            alphas.replace(CurrentN, fpos[CurrentN]),
            signs);
    }
};
//1D-specialization
template<int NumDimensions, typename Fetcher>
struct HypercubeInterpolator<NumDimensions, Fetcher, 0>
{
    __host__ __device__ static void interpolate(Fetcher& fetcher,
        const fposition_t<NumDimensions>& fpos, const iposition_t<NumDimensions>& iposL, const iposition_t<NumDimensions>& iposH, float alpha = 1)
    {
        //linear interpolation in 1D
        fetcher.fetchAndAddValue(iposL, alpha * (1 - fpos[0]));
        fetcher.fetchAndAddValue(iposL.replace(0, iposH[0]), alpha * fpos[0]);
    }

    template<int NumFeaturesPerLayer>
    __host__ __device__ static void adjoint(Fetcher& fetcher,
        const fposition_t<NumDimensions>& fpos, const iposition_t<NumDimensions>& iposL, const iposition_t<NumDimensions>& iposH,
        fposition_t<NumDimensions>& adjPosition, const feature_t<NumFeaturesPerLayer>& adjFeature,
        const fposition_t<NumDimensions>& alphas = fposition_t<NumDimensions>(1.f),
        const fposition_t<NumDimensions>& signs = fposition_t<NumDimensions>(1.f))
    {
        //fetch dot product
        float dot0 = fetcher.fetchAndDot(iposL, adjFeature);
        float dot1 = fetcher.fetchAndDot(iposL.replace(0, iposH[0]), adjFeature);

        //adjoint linear interpolation in 1D
        //last dimension
        float alphasWithout0 = alphas.reduceMulWithoutD(0, 1);
        adjPosition[0] += alphasWithout0 * (dot1 - dot0);
        //all other dimensions
        for (int d=1; d<NumDimensions; ++d)
        {
            float alphasWithoutD = alphas.reduceMulWithoutD(d, 1);
            float sign = signs[d];
            adjPosition[d] += sign * alphasWithoutD * ((1 - fpos[0]) * dot0 + fpos[0] * dot1);
        }
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

    static constexpr int NumOutputs = NumFeaturesPerLayer * (CombinationMode == LayerCombinationMode::CONCAT ? NumLayers : 1);

    /**
     * \brief Computes the forward code for a single layer
     * \tparam Add true iff the features should be added to the output, false if it should be assigned
     * \param position the position in the unit (hyper-)cube
     * \param cfg the layer configuration
     * \param parameters the parameter tensor
     * \param output the output (shared memory)
     */
    template<bool Add, typename O>
    static __device__ void forwardLayer(
        const fposition_t<NumDimensions>& position, const HashGridLayerConfig& cfg,
        const float* __restrict__ parameters, O& output, int outputOffset)
    {
        typedef typename O::ValueType O_t;

        //get corner positions and interpolation values
        fposition_t<NumDimensions> fpos;
        iposition_t<NumDimensions> iposL, iposH;
        const int rm1 = cfg.resolution - 1;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            float p = position[i] * (cfg.resolution - 1);
            iposL[i] = max(0, min(rm1, static_cast<int>(p)));
            iposH[i] = max(0, min(rm1, static_cast<int>(p)+1));
            fpos[i] = p - static_cast<float>(iposL[i]);

            //printf("[%03d] d=%d, iposL=%d, iposH=%d, fpos=%.3f\n",
            //    threadIdx.x, i, iposL[i], iposH[i], fpos[i]);
        }

        //interpolate feature
        feature_t<NumFeaturesPerLayer> feature{ zero_initialization_tag() };
        ValueFetcherForward<NumDimensions, NumFeaturesPerLayer> fetcher(cfg, parameters, feature);
        HypercubeInterpolator<NumDimensions, decltype(fetcher)>::interpolate(fetcher, fpos, iposL, iposH);

        //write to output
        if constexpr (Add)
        {
#pragma unroll
            for (int i = 0; i < NumFeaturesPerLayer; ++i) {
                output[i + outputOffset] += fcast<O_t>(feature[i]);
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NumFeaturesPerLayer; ++i) {
                //printf("[%03d] feature %d = %.4f\n", threadIdx.x, i, feature[i]);
                output[i + outputOffset] = fcast<O_t>(feature[i]);
            }
        }
    }

    template<typename I, typename O>
    static __device__ void forward(const I input, O& output, const param_t& params)
    {
        //fetch position and transform to unit cube
        fposition_t<NumDimensions> position;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            position[i] = (input[i + StartChannel] - params.boundingBoxMin[i]) * params.boundingBoxInvSize[i];
        }
        //printf("[%03d] unit position: %.3f, %.3f\n", threadIdx.x, position[0], position[1]);

        //loop over the layers
        if constexpr (CombinationMode == LayerCombinationMode::CONCAT)
        {
            for (int l = 0; l < NumLayers; ++l)
                forwardLayer<false>(position, params.layers[l], params.parametersForward, 
                    output, l * NumFeaturesPerLayer);
        }
        else if constexpr (CombinationMode == LayerCombinationMode::ADD)
        {
            forwardLayer<false>(position, params.layers[0], params.parametersForward, output, 0);
            for (int l = 1; l < NumLayers; ++l)
                forwardLayer<true>(position, params.layers[l], params.parametersForward, output, 0);
        }
    }

    /**
     * \brief Computes the forward code for a single layer
     * \param position the position in the unit (hyper-)cube
     * \param cfg the layer configuration
     * \param adjParameters the adjoint parameter tensor
     * \param adjOutput the output (shared memory)
     */
    template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename O>
    static __device__ void adjointLayer(
        const fposition_t<NumDimensions>& position, const HashGridLayerConfig& cfg,
        const float* __restrict__ parameters, float* adjParameters, const O& adjOutput, int outputOffset,
        fposition_t<NumDimensions>& adjPosition)
    {
        //get corner positions and interpolation values
        fposition_t<NumDimensions> fpos;
        iposition_t<NumDimensions> iposL, iposH;
        const int rm1 = cfg.resolution - 1;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            float p = position[i] * (cfg.resolution - 1);
            iposL[i] = max(0, min(rm1, static_cast<int>(p)));
            iposH[i] = max(0, min(rm1, static_cast<int>(p) + 1));
            fpos[i] = p - static_cast<float>(iposL[i]);

            //printf("[%03d] d=%d, iposL=%d, iposH=%d, fpos=%.3f\n",
            //    threadIdx.x, i, iposL[i], iposH[i], fpos[i]);
        }

        //fetch adjoint output
        feature_t<NumFeaturesPerLayer> adjFeature;
        for (int i = 0; i < NumFeaturesPerLayer; ++i)
            adjFeature[i] = fcast<float>(adjOutput[i + outputOffset]);

        //interpolate feature -> write adjoints to grid corners
        ValueFetcherAdjoint<NumDimensions, NumFeaturesPerLayer, EvaluateParameterGradients> fetcher(cfg, adjParameters, adjFeature);
        HypercubeInterpolator<NumDimensions, decltype(fetcher)>::interpolate(
            fetcher, fpos, iposL, iposH);

        //derivatives for the input position
        if constexpr(EvaluateInputGradients)
        {
            ValueFetcherDot<NumDimensions, NumFeaturesPerLayer> fetcherDot(cfg, parameters);
            HypercubeInterpolator<NumDimensions, decltype(fetcherDot)>::adjoint(
                fetcherDot, fpos, iposL, iposH, adjPosition, adjFeature);
        }
    }

    template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O, typename AdjI>
    static __device__ void adjoint(const I& input, const O& adjOutput, AdjI& adjInput, const param_t& params)
    {
        //fetch position and transform to unit cube
        fposition_t<NumDimensions> position;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            position[i] = (input[i + StartChannel] - params.boundingBoxMin[i]) * params.boundingBoxInvSize[i];
        }
        //printf("[%03d] unit position: %.3f, %.3f\n", threadIdx.x, position[0], position[1]);

        fposition_t<NumDimensions> adjPosition{zero_initialization_tag()};

        //loop over the layers
        if constexpr (CombinationMode == LayerCombinationMode::CONCAT)
        {
            for (int l = 0; l < NumLayers; ++l)
                adjointLayer<EvaluateInputGradients, EvaluateParameterGradients>(
                    position, params.layers[l], params.parametersForward, params.parametersBackward,
                    adjOutput, l * NumFeaturesPerLayer, adjPosition);
        }
        else if constexpr (CombinationMode == LayerCombinationMode::ADD)
        {
            for (int l = 0; l < NumLayers; ++l)
                adjointLayer<EvaluateInputGradients, EvaluateParameterGradients>(
                    position, params.layers[l], params.parametersForward, params.parametersBackward,
                    adjOutput, 0, adjPosition);
        }

        if constexpr (EvaluateInputGradients)
        {
            for (int i = 0; i < NumDimensions; ++i) {
                float adjP = adjPosition[i] * params.boundingBoxInvSize[i];
                //no atomic, position is only accessed by this thread
                adjInput[i + StartChannel] += adjP;
            }
        }
    }
};


QUICKMLP_KERNEL_NAMESPACE_END
