#include <qmlp/qmlp.h>
#include <spdlog/spdlog.h>

#include "spdlog/sinks/stdout_color_sinks.h"

QUICKMLP_NAMESPACE_BEGIN

QuickMLP::QuickMLP()
    : logger_(spdlog::stdout_color_mt("qmlp"))
    , kl_(std::make_shared<ckl::KernelLoader>(logger_))
#ifdef NDEBUG
    , enableCompileDebugMode_(false)
#else
    , enableCompileDebugMode_(true)
#endif
{
    kl_->setCacheDir(ckl::KernelLoader::DEFAULT_CACHE_DIR);
    //TODO: for now, use the file system
    //later, replace by embedded filesystem once the kernels are fixed.
    const std::filesystem::path current(__FILE__);
    const auto parent = current.parent_path().parent_path() / "include";
    kl_->setFileLoader(std::make_shared<ckl::FilesystemLoader>(parent));
}

QuickMLP& QuickMLP::Instance()
{
    static QuickMLP INSTANCE;
    return INSTANCE;
}

logger_t QuickMLP::getLogger() const
{
    return logger_;
}

void QuickMLP::setLogLevel(spdlog::level::level_enum level)
{
    logger_->set_level(level);
}

void QuickMLP::setLogLevel(const std::string& levelName)
{
    spdlog::level::level_enum level;
    if (levelName == "off")
        level = spdlog::level::off;
    else if (levelName == "debug")
        level = spdlog::level::debug;
    else if (levelName == "info")
        level = spdlog::level::info;
    else if (levelName == "warn")
        level = spdlog::level::warn;
    else if (levelName == "error")
        level = spdlog::level::err;
    else
        throw std::runtime_error("Unknown log level name '" + levelName + "', only 'off', 'debug', 'info', 'warn', 'error' are recognized");

    setLogLevel(level);
}

ckl::KernelLoader_ptr QuickMLP::kernelLoader() const
{
    return kl_;
}

void QuickMLP::setCompileDebugMode(bool enable)
{
    enableCompileDebugMode_ = enable;
}

int QuickMLP::getCompileFlags() const
{
    int compileFlags = ckl::KernelLoader::CompilationFlags::CompileThrowOnError;
    if (enableCompileDebugMode_)
    {
        compileFlags |= ckl::KernelLoader::CompilationFlags::CompileDebugMode;
    }
    return compileFlags;
}

QUICKMLP_NAMESPACE_END
