#pragma once

#include "common.h"
#include <ckl/errors.h>
#include <cuda_runtime.h>

QUICKMLP_NAMESPACE_BEGIN

/**
 * Helper class for RAII-style temporary memory.
 * If 'numBytes' is zero, no memory is allocated and
 * 'get' returns nullptr.
 */
class TmpMemory
{
    void* memory_;

public:
    TmpMemory(size_t numBytes): memory_(nullptr)
    {
        if (numBytes>0)
            CKL_SAFE_CALL(cudaMalloc(&memory_, numBytes));
    }
    ~TmpMemory()
    {
        CKL_SAFE_CALL_NO_THROW(cudaFree(memory_));
    }

    TmpMemory(const TmpMemory& other) = delete;
    TmpMemory(TmpMemory&& other) noexcept = delete;
    TmpMemory& operator=(const TmpMemory& other) = delete;
    TmpMemory& operator=(TmpMemory&& other) noexcept = delete;

    [[nodiscard]] void* get() { return memory_; }
    [[nodiscard]] const void* get() const { return memory_; }
};

QUICKMLP_NAMESPACE_END
