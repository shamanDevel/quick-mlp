#pragma once

#include <cuda_fp16.h>
#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

#include <qmlp/kernels/common.cuh>
#include <qmlp/kernels/encoding_line_integration_config.cuh>

QUICKMLP_KERNEL_NAMESPACE_BEGIN

template<int NumDimensions>
struct IntersectionRayAABB
{
    static __host__ __device__ __forceinline__ bool eval(const StaticArray<float, NumDimensions>& rayStart,
        const StaticArray<float, NumDimensions>& rayDir,
        const StaticArray<float, NumDimensions>& boxMin,
        const StaticArray<float, NumDimensions>& boxSize,
        float& tmin, float& tmax)
    {
        static_assert(NumDimensions != 3, "Bounding Box Intersection only implemented for 3D");
    }
};
template<>
struct IntersectionRayAABB<3>
{
    static __host__ __device__ __forceinline__ bool eval(const StaticArray<float, 3>& rayStart,
        const StaticArray<float, 3>& rayDir,
        const StaticArray<float, 3>& boxMin,
        const StaticArray<float, 3>& boxSize,
        float& tmin, float& tmax)
    {
        float invRayDir0 = 1.0f / rayDir[0];
        float invRayDir1 = 1.0f / rayDir[1];
        float invRayDir2 = 1.0f / rayDir[2];

        float t1 = (boxMin[0] - rayStart[0]) * invRayDir0;
        float t2 = (boxMin[0] + boxSize[0] - rayStart[0]) * invRayDir0;
        float t3 = (boxMin[1] - rayStart[1]) * invRayDir1;
        float t4 = (boxMin[1] + boxSize[1] - rayStart[1]) * invRayDir1;
        float t5 = (boxMin[2] - rayStart[2]) * invRayDir2;
        float t6 = (boxMin[2] + boxSize[2] - rayStart[2]) * invRayDir2;
        tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
        tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

        return tmax > tmin; //true iff intersection
    }
};

template<int StartChannel, int NumDimensions, LineIntegrationBlendingMode Blending, typename Child>
struct EncodingLineIntegration
{
    typedef LineIntegrationConfig<NumDimensions, typename Child::param_t> param_t;

    static constexpr int NumOutputs = Child::NumOutputs;
    typedef StaticArray<float, NumOutputs> feature_t;
    typedef StaticArray<float, NumDimensions> position_t;

    template<typename I, typename O>
    static __device__ void forward(const I input, O& output, const param_t& params)
    {
        //read ray start + ray dir
        position_t rayStart, rayDir;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            rayStart[i] = input[i + StartChannel];
            rayDir[i] = input[i + StartChannel + NumDimensions];
        }

        //compute bounding box intersection
        float tmin, tmax;
        bool intersection = IntersectionRayAABB<NumDimensions>::eval(
            rayStart, rayDir, params.boundingBoxMin, params.boundingBoxSize,
            tmin, tmax);

        const int steps = static_cast<int>(floorf((tmax-tmin)/params.stepsize));
        feature_t accu{ zero_initialization_tag() };
        //stepping
        for (float t = tmin; t < tmax; t+=params.stepsize)
        {
            //sample new contribution
            feature_t newContribution;
            position_t position = fma(t, rayDir, rayStart);
            Child::forward(position, newContribution, params.child);
            //blend
            accu += newContribution;
        }
        //blend 2
        if constexpr(Blending == LineIntegrationBlendingMode::AVERAGE)
        {
            if (steps > 0)
                accu *= (1.0f / static_cast<float>(steps));
        }

        //write output
#pragma unroll
        for (int i = 0; i < NumOutputs; ++i)
            output[i] = accu[i];
    }

    template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O, typename AdjI>
    static __device__ void adjoint(const I& input, const O& adjOutput, AdjI& adjInput, const param_t& params)
    {
        //read ray start + ray dir
        position_t rayStart, rayDir;
#pragma unroll
        for (int i = 0; i < NumDimensions; ++i) {
            rayStart[i] = input[i + StartChannel];
            rayDir[i] = input[i + StartChannel + NumDimensions];
        }

        //compute bounding box intersection
        float tmin, tmax;
        bool intersection = IntersectionRayAABB<NumDimensions>::eval(
            rayStart, rayDir, params.boundingBoxMin, params.boundingBoxSize,
            tmin, tmax);

        const int steps = static_cast<int>(floorf((tmax - tmin) / params.stepsize));

        //fetch adjoint output + adjoint blend 2
        float adjOutputScale;
        if constexpr (Blending == LineIntegrationBlendingMode::AVERAGE)
        {
            adjOutputScale = steps > 0 ? 1.0f / static_cast<float>(steps) : 1.0f;
        }
        else
            adjOutputScale = 1.0f;

        feature_t adjAccu;
#pragma unroll
        for (int i=0; i<NumOutputs; ++i)
        {
            adjAccu[i] = adjOutputScale * adjOutput[i];
        }

        position_t adjRayStart{ zero_initialization_tag() };
        position_t adjRayDir{ zero_initialization_tag() };

        //adjoint stepping
        for (float t = tmin; t < tmax; t += params.stepsize)
        {
            //adjoint: sample new contribution
            position_t position = fma(t, rayDir, rayStart); //t*rayDir + rayStart
            position_t adjPosition{zero_initialization_tag()};
            Child::template adjoint<EvaluateInputGradients, EvaluateParameterGradients>(
                position, adjAccu, adjPosition, params.child);
            //adjoint: position
            if constexpr (EvaluateInputGradients)
            {
                adjRayStart += adjPosition;
                adjRayDir += adjPosition*t;
            }
        }

        //adjoint: fetch ray start + direction
        if constexpr (EvaluateInputGradients)
        {
#pragma unroll
            for (int i = 0; i < NumDimensions; ++i) {
                adjInput[i + StartChannel] += adjRayStart[i];
                adjInput[i + StartChannel + NumDimensions] += adjRayDir[i];
            }
        }
    }
};

QUICKMLP_KERNEL_NAMESPACE_END
