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


};

QUICKMLP_NAMESPACE_END
