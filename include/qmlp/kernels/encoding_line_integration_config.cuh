#pragma once

#include "common.cuh"

QUICKMLP_KERNEL_NAMESPACE_BEGIN

/**
 * How are the features of the individual layers combined?
 */
enum class LineIntegrationBlendingMode
{
    /**
     * The samples are added together
     */
    ADDITIVE,
    /**
     * The samples are averaged.
     * If no sample is hit, zero is returned
     */
     AVERAGE
};

template<int NumDimensions, typename Child>
struct LineIntegrationConfig
{
    alignas(8) StaticArray<float, NumDimensions> boundingBoxMin;
    alignas(8) StaticArray<float, NumDimensions> boundingBoxSize;
    alignas(8) float stepsize;
    Child child;
};


QUICKMLP_KERNEL_NAMESPACE_END
