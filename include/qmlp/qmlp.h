#pragma once

#include "common.h"
#include <ckl/kernel_loader.h>

QUICKMLP_NAMESPACE_BEGIN

/**
 * The entry class into quick-mlp (singleton)
 */
class QuickMLP : public NonAssignable
{
private:
    ckl::KernelLoader_ptr kl_;
    bool enableDebugMode_;

    QuickMLP();

public:
    /**
     * Returns the global quick-mlp instance
     */
    static QuickMLP& Instance();

    /**
     * The kernel loader used to generate the cuda kernels.
     */
    [[nodiscard]] ckl::KernelLoader_ptr kernelLoader() const;

    /**
     * Enable (true) or disable (false) debug mode.
     * In debug mode, the verbose logging during compilation is enabled
     * and the kernels are also compiled with the debug flag.
     */
    void setDebugMode(bool enable);

    [[nodiscard]] bool isDebugMode() const { return enableDebugMode_; }

    /**
     * Returns the compile flags for CKL.
     */
    int getCompileFlags() const;
};

QUICKMLP_NAMESPACE_END
