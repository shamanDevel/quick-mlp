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
    bool enableCompileDebugMode_;
    bool enableVerboseLogging_;

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
     * Enable (true) or disable (false) debug compilation mode.
     * In debug mode, the kernels are compiled with the debug flag
     * and optimizations are disabled.
     *
     * Default setting: follows the debug build of the host code.
     */
    void setCompileDebugMode(bool enable);

    [[nodiscard]] bool isCompileDebugMode() const { return enableCompileDebugMode_; }

    /**
     * Enable (true) or disable (false) verbose logging.
     * This logs the sources of all compiled kernels.
     */
    void setVerboseLogging(bool enable);

    [[nodiscard]] bool isVerboseLogging() const { return enableVerboseLogging_; }

    /**
     * Returns the compile flags for CKL.
     */
    int getCompileFlags() const;
};

QUICKMLP_NAMESPACE_END
