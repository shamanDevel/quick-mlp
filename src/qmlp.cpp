#include <qmlp/qmlp.h>

QUICKMLP_NAMESPACE_BEGIN

QuickMLP::QuickMLP()
    : kl_(std::make_shared<ckl::KernelLoader>())
#ifdef NDEBUG
    , enableDebugMode_(false)
#else
    , enableDebugMode_(true)
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

ckl::KernelLoader_ptr QuickMLP::kernelLoader() const
{
    return kl_;
}

void QuickMLP::setDebugMode(bool enable)
{
    enableDebugMode_ = enable;
}

int QuickMLP::getCompileFlags() const
{
    int compileFlags = ckl::KernelLoader::CompilationFlags::CompileThrowOnError;
    if (enableDebugMode_)
    {
        compileFlags |= ckl::KernelLoader::CompilationFlags::CompileDebugMode
            | ckl::KernelLoader::CompilationFlags::CompileVerboseLogging;
    }
    return compileFlags;
}

QUICKMLP_NAMESPACE_END
