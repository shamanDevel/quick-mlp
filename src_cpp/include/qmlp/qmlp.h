#pragma once

#include "common.h"
#include <ckl/kernel_loader.h>
#include <spdlog/fwd.h>

QUICKMLP_NAMESPACE_BEGIN

typedef std::shared_ptr<spdlog::logger> logger_t;

/**
 * The entry class into quick-mlp (singleton)
 */
class QuickMLP : public NonAssignable
{
private:
    static std::unique_ptr<QuickMLP> INSTANCE;

    const logger_t logger_;
    const ckl::KernelLoader_ptr kl_;
    bool enableCompileDebugMode_;

    QuickMLP();

public:
    /**
     * Returns the global quick-mlp instance
     */
    static QuickMLP& Instance();

    /**
     * Deletes the global quick-mlp instance
     */
    static void DeleteInstance();

    /**
     * Tests if CUDA is available.
     * If this is false, all kernel calls will fail with an exception
     * \return true iff CUDA is available.
     */
    [[nodiscard]] bool isCudaAvailable() const;

    /**
     * Tests if CUDA is available and if not, throw a ckl::cuda_error.
     */
    void checkCudaAvailable() const;

    /**
     * Returns the logger instance used to report compile logs (debug) or errors
     */
    [[nodiscard]] logger_t getLogger() const;

    /**
     * Sets the log level.
     * The kernel loader uses the following levels:
     *  - debug: verbose info on the kernel names and source code
     *  - info: a new kernel is compiled
     *  - error: compilation errors
     */
    void setLogLevel(spdlog::level::level_enum level);

    /**
     * Sets the log level by name
     * The kernel loader uses the following levels:
     *  - debug: verbose info on the kernel names and source code
     *  - info: a new kernel is compiled
     *  - error: compilation errors
     *
     * \param levelName the name of the level: "off", "debug", "info", "warn", "error"
     */
    void setLogLevel(const std::string& levelName);

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
     * Returns the compile flags for CKL.
     */
    int getCompileFlags() const;
};

QUICKMLP_NAMESPACE_END
