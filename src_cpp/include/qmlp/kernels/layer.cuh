#pragma once

#ifndef DEBUG_PRINT
//Set to 1 to enable very verbose debug messages
#define DEBUG_PRINT 0
#endif

#include <cuda_fp16.h>
#include <mma.h>

#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#include <stdio.h>
#endif

#include <qmlp/kernels/common.cuh>

QUICKMLP_KERNEL_NAMESPACE_BEGIN

//DEBUG FUNCTIONS

namespace detail
{
    __forceinline__ __host__ __device__ bool isNaN(float v) { return ::isnan(v); }
    __forceinline__ __host__ __device__ bool isNaN(half v) { return ::__hisnan(v); }
}

template<typename Fragment>
__device__ void assertFragmentNotNaN(const Fragment& f, const char* msg)
{
    __syncwarp();
    for (int t = 0; t < f.num_elements; t++)
    {
#if DEBUG_PRINT==1
        if (detail::isNaN(f.x[t])) printf("[%04d] NaN at fragment entry %d (%s)\n", threadIdx.x, t, msg);
#else
        assert(!detail::isNaN(f.x[t]));
#endif
    }
    __syncwarp();
}


static inline __device__
void printLayer(int layer, int idx, const half* data, int numel)
{
    int i = 0;
    //printf("{L %d}[T %03d] access memory from 0x%llx to 0x%llx\n",
    //    layer, idx, reinterpret_cast<long long>(data), reinterpret_cast<long long>(data + numel));
    while (numel > 0)
    {
        printf("{L %d}[T %03d] (%02d-%02d): %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
            layer, idx, i, i + 16,
            __half2float(data[i + 0]), __half2float(data[i + 1]), __half2float(data[i + 2]), __half2float(data[i + 3]),
            __half2float(data[i + 4]), __half2float(data[i + 5]), __half2float(data[i + 6]), __half2float(data[i + 7]),
            __half2float(data[i + 8]), __half2float(data[i + 9]), __half2float(data[i + 10]), __half2float(data[i + 11]),
            __half2float(data[i + 12]), __half2float(data[i + 13]), __half2float(data[i + 14]), __half2float(data[i + 15]));

        i += 16;
        numel -= 16;
    }
}
static inline __device__
void printLayerBinary(int layer, int idx, const half* data, int numel)
{
    const unsigned short* dataS = reinterpret_cast<const unsigned short*>(data);
    int i = 0;
    //printf("{L %d}[T %03d] access memory from 0x%llx to 0x%llx\n",
    //    layer, idx, reinterpret_cast<long long>(data), reinterpret_cast<long long>(data + numel));
    while (numel > 0)
    {
        printf("{L %d}[T %03d] (%02d-%02d): %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx %04hx\n",
            layer, idx, i, i + 16,
            (dataS[i + 0]), (dataS[i + 1]), (dataS[i + 2]), (dataS[i + 3]),
            (dataS[i + 4]), (dataS[i + 5]), (dataS[i + 6]), (dataS[i + 7]),
            (dataS[i + 8]), (dataS[i + 9]), (dataS[i + 10]), (dataS[i + 11]),
            (dataS[i + 12]), (dataS[i + 13]), (dataS[i + 14]), (dataS[i + 15]));

        i += 16;
        numel -= 16;
    }
}

// MAIN LAYER COMPUTATION

template<int InChannelsDiv16, int OutChannelsDiv16, int HiddenStride, bool Bias, typename Activation>
struct Layer
{
    static constexpr int MaxChannelsDiv16 = InChannelsDiv16 > OutChannelsDiv16 ? InChannelsDiv16 : OutChannelsDiv16;
    static constexpr int MaxChannels = MaxChannelsDiv16 * 16;
    static constexpr int InChannels = InChannelsDiv16 * 16;
    static constexpr int OutChannels = OutChannelsDiv16 * 16;

    /**
     * Performs inference
     * \param weights the pointer to the parameter of the weight matrix. Shared or global memory
     * \param bias the pointer to the bias matrix or nullptr if Bias=false. Shared or global memory
     * \param statesInout pointer to the intermediate states, input and output in shared memory (per warp). Stride=MaxChannelDiv16*16.
     * \param intermediateResults [StoreIntermediateResults=true] the per-warp intermediate results
     */
    template<bool StoreIntermediateResults>
    __device__ static void inference(const half* weights, const half* bias, half* statesInout, half* intermediateResults)
    {
        using namespace nvcuda::wmma;
        //weights
        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[OutChannelsDiv16][InChannelsDiv16]; //row,col
        //inputs
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[InChannelsDiv16][2];
        //outputs
        fragment<accumulator, 16, 16, 16, half> c_frag[OutChannelsDiv16][2];

        if constexpr(Bias)
        {
            //load bias
            for (int cout = 0; cout < OutChannelsDiv16; ++cout)
            {
                load_matrix_sync(c_frag[cout][0], bias + 16 * cout, 0, mem_col_major);
                load_matrix_sync(c_frag[cout][1], bias + 16 * cout, 0, mem_col_major);
            }
        }
        else
        {
            //initialize with zeros
            for (int cout = 0; cout < OutChannelsDiv16; ++cout)
            {
                fill_fragment(c_frag[cout][0], hZERO());
                fill_fragment(c_frag[cout][1], hZERO());
            }
        }

        //load weights (A)
        for (int cout = 0; cout < OutChannelsDiv16; ++cout)
            for (int cin = 0; cin < InChannelsDiv16; ++cin)
                load_matrix_sync(a_frag[cout][cin],
                    weights + 16 * (cin + InChannels * cout),
                    InChannels);

        //load input (B)
        for (int cin = 0; cin < InChannelsDiv16; ++cin)
        {
            load_matrix_sync(b_frag[cin][0], statesInout + 16 * cin, HiddenStride);
            load_matrix_sync(b_frag[cin][1], statesInout + 16 * cin + 16 * HiddenStride, HiddenStride);
        }

        //matmul
        for (int i = 0; i < OutChannelsDiv16; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < InChannelsDiv16; ++k) {
                    mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
                }
            }
        }

        //for a forward pass with checkpoints,
        //store the states now
        if constexpr(StoreIntermediateResults)
        {
            for (int cout = 0; cout < OutChannelsDiv16; ++cout)
            {
                store_matrix_sync(intermediateResults + 16 * cout, c_frag[cout][0], OutChannels, mem_col_major);
                store_matrix_sync(intermediateResults + 16 * cout + 16 * OutChannels, c_frag[cout][1], OutChannels, mem_col_major);
            }
        }

        //activations
        for (int i = 0; i < OutChannelsDiv16; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int t = 0; t < c_frag[0][0].num_elements; t++)
                {
                    c_frag[i][j].x[t] = Activation::forward(c_frag[i][j].x[t]);
                }
            }
        }

        //copy to shared
        for (int cout = 0; cout < OutChannelsDiv16; ++cout)
        {
            store_matrix_sync(statesInout + 16 * cout, c_frag[cout][0], HiddenStride, mem_col_major);
            store_matrix_sync(statesInout + 16 * cout + 16 * HiddenStride, c_frag[cout][1], HiddenStride, mem_col_major);
        }
    }

    /**
     * \brief Adjoint pass
     * \tparam ComputeWeightGradients 
     * \param weights the weight matrix of shape (Cout, Cin); shared or global memory
     * \param bias the bias vector shape (Cout); shared or global memory
     * \param adjStatesInout the adjoint intermediate states (HiddenStride, 32); shared
     * \param intermediateResultsIn the intermediate results from the forward pass, (Cout, 32); global; column-major
     * \param adjIntermediateOut the adjoints of the intermediate results for
     *    optimizing the weights and bias, shape (Cout, 32); global; column-major.
     *    Only set if ComputeWeightGradients=true
     */
    template<bool ComputeWeightGradients>
    __device__ static void adjoint(const half* weights, const half* bias, half* adjStatesInout, 
        const half* intermediateResultsIn, half* adjIntermediateOut)
    {
        using namespace nvcuda::wmma;
        //weights
        fragment<matrix_a, 16, 16, 16, half, col_major> a_frag[InChannelsDiv16][OutChannelsDiv16]; //row,col
        //inputs
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[OutChannelsDiv16][2];
        //outputs
        fragment<accumulator, 16, 16, 16, half> c_frag[InChannelsDiv16][2];

        //load weights (A) transposed!
        for (int cout = 0; cout < OutChannelsDiv16; ++cout)
            for (int cin = 0; cin < InChannelsDiv16; ++cin)
                load_matrix_sync(a_frag[cin][cout],
                    weights + 16 * (cin + InChannels * cout),
                    InChannels);

        //adjoint of the activation.
        //Perform computation in shared memory, directly in adjStatesInout,
        //Use the inputs from intermediateResultsIn.
        const int lineID = threadIdx.x % 32;
        __syncwarp();
        for (int cout=0; cout<OutChannels; ++cout)
        {
            half activationInput = intermediateResultsIn[cout + OutChannels * lineID];
            half adjActivationOutput = adjStatesInout[cout + HiddenStride * lineID];
            half adjActivationInput = Activation::adjoint(activationInput, adjActivationOutput);
            adjStatesInout[cout + HiddenStride * lineID] = adjActivationInput;
            //Write the output to adjIntermediateOut
            if constexpr (ComputeWeightGradients)
            {
                adjIntermediateOut[cout + OutChannels * lineID] = adjActivationInput;
            }
        }
        __syncwarp(); //make changes to shared memory (activations) visible

#if DEBUG_PRINT==1
        //TEST
        if constexpr (ComputeWeightGradients)
            printLayer(0, threadIdx.x, adjIntermediateOut + (OutChannels * lineID), OutChannels);
#endif

        //load it into fragment b_frag
        for (int cout = 0; cout < OutChannelsDiv16; ++cout)
        {
            load_matrix_sync(b_frag[cout][0], adjStatesInout + 16 * cout, HiddenStride);
            load_matrix_sync(b_frag[cout][1], adjStatesInout + 16 * cout + 16 * HiddenStride, HiddenStride);
        }

        //compute adjoint of the input
        for (int cin = 0; cin < InChannelsDiv16; ++cin)
        {
            fill_fragment(c_frag[cin][0], hZERO());
            fill_fragment(c_frag[cin][1], hZERO());
        }
        for (int i = 0; i < InChannelsDiv16; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < OutChannelsDiv16; ++k) {
                    mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
                }
            }
        }

        //copy to shared
        for (int cin = 0; cin < InChannelsDiv16; ++cin)
        {
            store_matrix_sync(adjStatesInout + 16 * cin, c_frag[cin][0], HiddenStride, mem_col_major);
            store_matrix_sync(adjStatesInout + 16 * cin + 16 * HiddenStride, c_frag[cin][1], HiddenStride, mem_col_major);
        }
    }
};

QUICKMLP_KERNEL_NAMESPACE_END
