#pragma once

/*
 * This is the template for the weight update (gradient computation).
 * Modified by the host code before compiled,
 * everything in $$..$$ will be replaced
 */

#ifndef DEBUG_PRINT
 //Set to 1 to enable very verbose debug messages and assertions
#define DEBUG_PRINT 0
#endif

#ifndef CUDA_NO_HOST
#include "assert.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#endif
#include <cuda_fp16.h>

#include <qmlp/kernels/common.cuh>
#include <qmlp/kernels/layer.cuh>
#include <qmlp/kernels/tensor.cuh>
#include <qmlp/kernels/loops.cuh>

$$INCLUDES$$

$$ENCODING_CONSTANTS$$


QUICKMLP_KERNEL_NAMESPACE_BEGIN

namespace activations
{
    $$DEFINE_ACTIVATIONS$$
}


/**
 * \brief Loader for loading hat{O}_tmp into shared memory and registers.
 * This is used as the A-matrix for the matmul.
 */
template<int _MDiv16>
struct OHatTmpLoader
{
    /**
     * Input datatype for loading from global memory
     */
    typedef half* input_t;

    /**
     * Layout of the matrix in memory.
     * This matches the layout of \c adjIntermediateOut in \c Layer::adjoint
     */
    typedef nvcuda::wmma::col_major layout_t;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, layout_t> fragment_t;
    static constexpr int MDiv16 = _MDiv16;
    static constexpr int M = MDiv16 * 16;

    static constexpr int SharedMemoryHalf = 32 * M;

    /**
     * Loads the current block from global memory \c srcGlobal into shared memory
     * \c tmpShared. The per-warp shared memory has \ref SharedMemoryHalf entries
     *
     * \param numel the number of active elements. If <32, the warp is not fully active
     */
    __device__ static void loadToShared(half* tmpShared, const input_t srcGlobal, int warpID, int numel)
    {
        //always load a full warp of 32 values.
        //The network evaluation (forward+backward) pads half-filled last warps with zeros

        //CUDA can read 4 bytes at once -> use half2
        const half2* src = reinterpret_cast<const half2*>(srcGlobal + warpID*32*M);
        half2* dst = reinterpret_cast<half2*>(tmpShared);

#if DEBUG_PRINT==1
        //TEST
        printLayer(0, threadIdx.x, srcGlobal + (warpID * 32 * M) + (M * (threadIdx.x % 32)), M);
#endif

        //now load 32*M entries, linear in memory. Layout does not matter here
        const int lineID = threadIdx.x % 32;
        static constexpr int M2 = MDiv16 * 8; //number of half2 entries
#pragma unroll
        for (int cout = 0; cout < M2; ++cout)
        {
            //TODO: optimize with __ldcs(const T* address)
            dst[32 * cout + lineID] = src[32 * cout + lineID];
        }
        __syncwarp(); //make it visible across the warp for load_matrix_sync
    }

    /**
     * Loads the current block from shared memory \c tmpShared into the
     * wmma::fragments for the tensor core multiplication
     */
    __device__ static void loadToFragment(fragment_t dst[MDiv16][2], const half* tmpShared)
    {
#if DEBUG_PRINT==1
        //TEST
        const int lineID = threadIdx.x % 32;
        //printLayerBinary(10, threadIdx.x, tmpShared + (M * lineID), M);
        printLayer(10, threadIdx.x, tmpShared + (M * lineID), M);
#endif

#pragma unroll
        for (int cout = 0; cout < MDiv16; ++cout)
        {
            using namespace nvcuda::wmma;
            load_matrix_sync(dst[cout][0], tmpShared + 16 * cout, M);
            load_matrix_sync(dst[cout][1], tmpShared + 16 * cout + 16 * M, M);
        }
    }
};

/**
 * \brief Loader for loading I^T into shared memory and registers.
 * The input I^T is not available directly, only the pre-activation output of the previous
 * layout. Therefore, run the activation functions again .
 * Also, the memory layout in global memory is column major. But we need a transposed
 * fragment, hence treat it as row major during loading.
 */
template<int _NDiv16, typename Activation>
struct HiddenLoader
{
    /**
     * Input datatype for loading from global memory
     */
    typedef half* input_t;

    /**
     * Layout of the matrix in memory.
     * This is the transpose of the forward temporary memory, that is
     * stored as col-major in Layer::inference
     */
    typedef nvcuda::wmma::row_major layout_t;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, layout_t> fragment_t;
    static constexpr int NDiv16 = _NDiv16;
    static constexpr int N = NDiv16 * 16;

    static constexpr int SharedMemoryHalf = 32 * N;

    /**
     * Loads the current block from global memory \c srcGlobal into shared memory
     * \c tmpShared. The per-warp shared memory has \ref SharedMemoryHalf entries
     */
    __device__ static void loadToShared(half* tmpShared, const input_t srcGlobal, int warpID, int numel)
    {
        //always load a full warp of 32 values.
        //The network evaluation (forward+backward) pads half-filled last warps with zeros

        //CUDA can read 4 bytes at once -> use half2
        const half2* src = reinterpret_cast<const half2*>(srcGlobal + warpID*32*N);
        half2* dst = reinterpret_cast<half2*>(tmpShared);

        //now load 32*M entries, linear in memory. Layout does not matter here
        __syncwarp();
        const int lineID = threadIdx.x % 32;
        static constexpr int N2 = NDiv16 * 8; //number of half2 entries
#pragma unroll
        for (int cin = 0; cin < N2; ++cin)
        {
            //TODO: optimize with __ldcs(const T* address)
            dst[32 * cin + lineID] = src[32 * cin + lineID];
        }
        __syncwarp(); //make it visible across the warp for load_matrix_sync
    }

    /**
     * Loads the current block from shared memory \c tmpShared into the
     * wmma::fragments for the tensor core multiplication
     */
    __device__ static void loadToFragment(fragment_t dst[2][NDiv16], const half* tmpShared)
    {
#if DEBUG_PRINT==1
        //TEST
        printLayer(1, threadIdx.x, tmpShared + (N*(threadIdx.x%32)), N);
#endif

        //note: load as row-major for transposing!
#pragma unroll
        for (int cin = 0; cin < NDiv16; ++cin)
        {
            //note: load as row-major for transposing!
            assert(isAligned<8>(tmpShared + 16 * cin));
            assert(isAligned<8>(tmpShared + 16 * cin + 16 * N));

            using namespace nvcuda::wmma;
            load_matrix_sync(dst[0][cin], tmpShared + 16 * cin, N);
            load_matrix_sync(dst[1][cin], tmpShared + 16 * cin + 16 * N, N);
        }
        //run the activations again
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < NDiv16; ++i) {
                for (int t = 0; t < dst[0][0].num_elements; t++)
                {
                    dst[j][i].x[t] = Activation::forward(dst[i][j].x[t]);
                }
            }
        }
    }
};

/**
 * \brief Loader for loading I^T into shared memory and registers.
 * First, the input encodings are run again and the results stored in shared memory.
 * Then, the input matrix is transposed during loading into registers.
 */
template<int _NDiv16>
struct InputLoader
{
    /**
     * Input datatype for loading from global memory
     */
    typedef Tensor2Read<float> input_t;

    /**
     * Layout of the matrix in shared memory.
     */
    typedef nvcuda::wmma::row_major layout_t;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, layout_t> fragment_t;
    static constexpr int NDiv16 = _NDiv16;
    static constexpr int N = NDiv16 * 16;

    static constexpr int SharedMemoryHalf = 32 * N;

    /**
     * Loads the current block from global memory \c srcGlobal into shared memory
     * \c tmpShared. The per-warp shared memory has \ref SharedMemoryHalf entries
     */
    __device__ static void loadToShared(half* tmpShared, const input_t srcGlobal, int warpID, int numel)
    {
        const int lineID = threadIdx.x % 32;
        const bool valid = lineID < numel;

        constexpr int INPUT_PAD_START = $$INPUT_PAD_START$$;
        constexpr int CHANNELS_IN = $$CHANNELS_IN$$;
        static_assert(CHANNELS_IN == N, "template parameter N does not match CHANNELS_IN");

        half* intermediateResultsThread = tmpShared + N * lineID;
        const int index = 32 * warpID + lineID;
        if (valid)
        {
            auto encodingInput = srcGlobal[index];
            WrappedArray<half> encodingOutput{ intermediateResultsThread, N };

            //CODE GENERATION [[
$$CALL_ENCODINGS$$
            //]] CODE GENERATIION

            //padding
            for (int cin = INPUT_PAD_START; cin < CHANNELS_IN; ++cin)
            {
                intermediateResultsThread[cin] = hZERO();
            }
        }
        else
        {
            //invalid index, fill with zeros to avoid NaNs
            for (int cin = 0; cin < CHANNELS_IN; ++cin)
            {
                intermediateResultsThread[cin] = hZERO();
            }
        }

#if DEBUG_PRINT==1
        //TEST
        __syncwarp();
        for (int cin=0; cin<N; ++cin)
        {
            half v = intermediateResultsThread[cin];
            if (detail::isNaN(v))
                printf("NaN at index=%d, cin=%d\n", index, cin);
        }
#endif

        __syncwarp(); //make the shared memory visible to the warp for load_matrix_sync
    }

    /**
     * Loads the current block from shared memory \c tmpShared into the
     * wmma::fragments for the tensor core multiplication
     */
    __forceinline__ __device__ static void loadToFragment(fragment_t dst[2][NDiv16], const half* tmpShared)
    {
        /*
         * tmpShared:
         *  - channel from 0 to N-1
         *  - batch from 0 to 31
         * matrix in shared memory: channel rows, batch columns, column major
         *    element(channel, batch) = tmpShared[channel + N*batch]
         * We need it transposed as a matrix of shape (batch, channels) =: I'
         *    -> load it row-major
         *    fragment[0, cin] spans I'[0:16, 16*cin:16*(cin+1)]
         *       = tmpShared with offset 16*cin and stride N
         *    fragment[1, cin] spans I'[16:32, 16*cin:16*(cin+1)]
         *       = tmpShared with offset 16*cin+16*N and stride N
         */

#if DEBUG_PRINT==1
        //TEST
        const int lineID = threadIdx.x % 32;
        printLayer(11, threadIdx.x, tmpShared + (N*lineID), N);
        printf("[%04d] tmpShared=0x%llx, N=%d\n", threadIdx.x, reinterpret_cast<unsigned long long>(tmpShared), N);
#endif
        
#pragma unroll
        for (int cin = 0; cin < NDiv16; ++cin)
        {
            //note: load as row-major for transposing!
            assert(isAligned<8>(tmpShared + 16 * cin));
            assert(isAligned<8>(tmpShared + 16 * cin + 16 * N));

            using namespace nvcuda::wmma;
            load_matrix_sync(dst[0][cin], tmpShared + 16 * cin, N);
            load_matrix_sync(dst[1][cin], tmpShared + 16 * cin + 16*N, N);

#if DEBUG_PRINT==1
            assertFragmentNotNaN(dst[0][cin], "load0");
            assertFragmentNotNaN(dst[1][cin], "load1");
#endif
        }
    }
};

/**
 * \brief Computes C += A*B^T block-synchronous.
 * The matrix A has shape MxK, matrix B has shape NxK.
 * M and N must be multiples of 16 and compile-time constants, specified by MDiv16, NDiv16.
 * K is flexible and can be large.
 *
 * TODO: Bias reduction
 *
 * \tparam AccuT the type of C, can be half or float
 * \tparam ALoader the loader for matrix A of shape M*K (OHatLoader)
 * \tparam BLoader the loader for matrix B of shape K*N (HiddenLoader or InputLoader)
 * \param numel the number of active elements
 * \param outAdjWeights the adjoint of weight matrix, row-major
 */
template<typename AccuT, typename ALoader, typename BLoader,
    typename AInput = typename ALoader::input_t,
    typename BInput = typename BLoader::input_t>
__global__ void WeightUpdateSingleBlockKernel(
    int numel, AccuT* outAdjWeights, AInput aIn, BInput bIn)
{
    const int warpID = threadIdx.x / 32;
    const int lineID = threadIdx.x % 32;
    const int numWarps = blockDim.x / 32;
    assert(gridDim.x == 1); //only a single block support (no atomics / grid-wise reduction)

    constexpr int MDiv16 = ALoader::MDiv16;
    constexpr int NDiv16 = BLoader::NDiv16;
    constexpr int M = MDiv16 * 16;
    constexpr int N = NDiv16 * 16;
    typedef typename ALoader::fragment_t AFragment_t;
    typedef typename BLoader::fragment_t BFragment_t;

    using namespace nvcuda::wmma;

    //per-warp: store the result of matrix C, shape MxN
    fragment<accumulator, 16, 16, 16, AccuT> c_frag[MDiv16][NDiv16];
    for (int m=0; m<MDiv16; ++m) for (int n=0; n<NDiv16; ++n)
    {
        fill_fragment(c_frag[m][n], hZERO());
    }

    // //bias
    // StaticArray<AccuT, MDiv16> bias_frag;

    //matrices A and B
    constexpr int SharedBytesPerWarp_Input = //storage for A,B
        sizeof(half) * (
        ALoader::SharedMemoryHalf +
        BLoader::SharedMemoryHalf);
    constexpr int SharedBytesPerWarp_Output = //storage for C
        sizeof(AccuT) * M * N;

    //This is the memory that must be provided by the host code:
    [[maybe_unused]]
    constexpr int SharedBytesPerWarp = max(SharedBytesPerWarp_Input, SharedBytesPerWarp_Output);
    if (threadIdx.x == 0)
        printf("SharedBytesPerWarp_Input=%d, SharedBytesPerWarp_Output=%d\n",
            SharedBytesPerWarp_Input, SharedBytesPerWarp_Output);

    extern __shared__ char sIntermediate[];

    half* intermediateWarp_Input = reinterpret_cast<half*>(sIntermediate + (SharedBytesPerWarp_Input * warpID));
    AFragment_t a_frag[MDiv16][2];
    BFragment_t b_frag[2][NDiv16];

    //now loop over the partitions of K of size 32
    const auto warps_pow32 = divRoundUp(numel, 32);
    if (threadIdx.x == 0) printf("active warps=%d, numWarps=%d\n", warps_pow32, numWarps);
    for (int warpIndex = warpID; 
        warpIndex < warps_pow32;
        warpIndex += numWarps)
    {

        //the logical index into the arrays (not needed)
        [[maybe_unused]]
        int elementIndex = warpIndex + lineID;

        //the remaining elements in the array.
        //If this values is <32, this warp is only half filled,
        //and special care must be taken for loading+saving
        int elementsLeft = numel - warpIndex * 32;
        
        //load A and B to shared
        half* aShared = intermediateWarp_Input;
        half* bShared = intermediateWarp_Input + ALoader::SharedMemoryHalf;
        ALoader::loadToShared(aShared, aIn, warpIndex, elementsLeft);
        BLoader::loadToShared(bShared, bIn, warpIndex, elementsLeft);

        //load to fragments
        ALoader::loadToFragment(a_frag, aShared);
        BLoader::loadToFragment(b_frag, bShared);

#if DEBUG_PRINT==1
        //Test
        for (int m = 0; m < MDiv16; ++m) for (int k = 0; k < 2; ++k)
            assertFragmentNotNaN(a_frag[m][k], "a_frag");
        for (int n = 0; n < NDiv16; ++n) for (int k = 0; k < 2; ++k)
            assertFragmentNotNaN(b_frag[k][n], "b_frag");
#endif

        //matmul, accumulates in the per-warp c-matrix
        for (int k = 0; k < 2; ++k)
        {
            for (int m = 0; m < MDiv16; ++m) for (int n = 0; n < NDiv16; ++n)
            {
                mma_sync(c_frag[m][n], a_frag[m][k], b_frag[k][n], c_frag[m][n]);
            }
        }

#if DEBUG_PRINT==1
        //Test
        for (int m = 0; m < MDiv16; ++m) for (int n = 0; n < NDiv16; ++n)
            assertFragmentNotNaN(c_frag[m][n], "c_frag");
#endif
    }

    __syncthreads(); //let's wait until all warps are finished.
                     //Otherwise, we could overwrite the tmpShared of other warps

    //reduce C across warps
    //but first, we need to store it to shared memory
    AccuT* cSharedWarp = reinterpret_cast<AccuT*>(sIntermediate + SharedBytesPerWarp_Output * warpID);
    for (int m = 0; m < MDiv16; ++m) for (int n = 0; n < NDiv16; ++n)
    {
        store_matrix_sync(cSharedWarp + 16 * n + N * 16 * m, c_frag[m][n], N, mem_row_major);
    }

    //now perform the reduction over the whole block
    __syncthreads();
    AccuT* cSharedBlock = reinterpret_cast<AccuT*>(sIntermediate);
    constexpr int reduceBatches = M * N;
    const int reduceElements = numWarps;
    for (int i=threadIdx.x; i<reduceBatches; i+=blockDim.x)
    {
        AccuT accu = cSharedBlock[i];
//#if DEBUG_PRINT==1
//        printf("cShared[%03d,000]=%.4f\n", i, accu);
//#endif
        for (int j = 1; j < reduceElements; ++j) {
            accu += cSharedBlock[i + j * reduceBatches];
//#if DEBUG_PRINT==1
//            printf("cShared[%03d,%03d]=%.4f\n", i, j, cSharedBlock[i + j * reduceBatches]);
//#endif
        }
        //write the matrix back to global memory
        outAdjWeights[i] += accu;
    }
}

QUICKMLP_KERNEL_NAMESPACE_END

