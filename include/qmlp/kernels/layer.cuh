#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#ifndef CUDA_NO_HOST
#include <cuda_runtime.h>
#endif

namespace qmlp { namespace kernel {

template<int InChannelsDiv16, int OutChannelsDiv16, bool Bias, typename Activation>
struct Layer
{
    static constexpr int MaxChannelsDiv16 = InChannelsDiv16 > OutChannelsDiv16 ? InChannelsDiv16 : OutChannelsDiv16;
    static constexpr int MaxChannels = MaxChannelsDiv16 * 16;
    static constexpr int InChannels = InChannelsDiv16 * 16;
    static constexpr int OutChannels = OutChannelsDiv16 * 16;

    /**
     * Performs inference (forward pass without storing the intermediate results).
     * \param weights the pointer to the parameter of the weight matrix. Shared or global memory
     * \param bias the pointer to the bias matrix or nullptr if Bias=false. Shared or global memory
     * \param statesInout pointer to the intermediate states, input and output in shared memory. Stride=MaxChannelDiv16*16.
     */
    static void inference(const half* weights, const half* bias, half* statesInout)
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
                fill_fragment(c_frag[cout][0], __float2half(0));
                fill_fragment(c_frag[cout][1], __float2half(0));
            }
        }

        //load weights (A)
        for (int cout = 0; cout < OutChannelsDiv16; ++cout)
            for (int cin = 0; cin < InChannelsDiv16; ++cin)
                load_matrix_sync(a_frag[cout][cin],
                    weights + 16 * cin + OutChannels * cout,
                    InChannels);

        //load input (B)
        for (int cin = 0; cin < InChannelsDiv16; ++cin)
        {
            load_matrix_sync(b_frag[cin][0], statesInout + 16 * cin, MaxChannels);
            load_matrix_sync(b_frag[cin][1], statesInout + 16 * cin + 16 * InChannels, MaxChannels);
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
        //Here would be the storage to global memory

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
            store_matrix_sync(statesInout + 16 * cout, c_frag[cout][0], MaxChannels, mem_col_major);
            store_matrix_sync(statesInout + 16 * cout + 16 * OutChannels, c_frag[cout][1], MaxChannels, mem_col_major);
        }
    }
};



}}
