#include "../qmlp/bindings.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <spdlog/spdlog.h>

#include <qmlp/activation.h>
#include <qmlp/qmlp.h>

#include "bind_activation.h"
#include "bind_encoding.h"
#include "bind_network.h"
#include "bind_utils.h"

QUICKMLP_NAMESPACE_BEGIN

QUICKMLP_NAMESPACE::Tensor wrap(const torch::Tensor& t)
{
    if (!t.defined())
    {
        //attempt to wrap an undefined tensor
        return {};
    }
    TORCH_CHECK(t.is_cuda(), "Tensor must reside on the GPU");

    qmlp::Tensor::Precision p;
    if (t.dtype() == c10::kFloat)
        p = qmlp::Tensor::FLOAT;
    else if (t.dtype() == c10::kHalf)
        p = qmlp::Tensor::HALF;
    else
    {
        TORCH_CHECK(false, "Unsupported datatype, only float and half tensors supported, but is ", t.dtype());
    }

    std::vector<int32_t> sizes(t.sizes().begin(), t.sizes().end());
    std::vector<int32_t> strides(t.strides().begin(), t.strides().end());

    return QUICKMLP_NAMESPACE::Tensor( t.data_ptr(), p, sizes, strides );
}

torch::Tensor unwrap(QUICKMLP_NAMESPACE::Tensor& t)
{
    std::vector<int64_t> sizes(t.sizes().begin(), t.sizes().end());
    torch::IntArrayRef sizesTorch(sizes);

    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
    torch::IntArrayRef stridesTorch(strides);

    c10::ScalarType dtype;
    if (t.precision() == Tensor::HALF)
        dtype = c10::kHalf;
    else if (t.precision() == Tensor::FLOAT)
        dtype = c10::kFloat;
    else //double
        dtype = c10::kDouble;

    return torch::from_blob(t.dataPtr<void>(), sizesTorch, stridesTorch,
        at::TensorOptions().device(c10::kCUDA).dtype(dtype));
}

c10::ScalarType unwrapDtype(qmlp::Tensor::Precision p)
{
    switch (p)
    {
    case Tensor::DOUBLE: return c10::kDouble;
    case Tensor::FLOAT: return c10::kFloat;
    case Tensor::HALF: return c10::kHalf;
    default: return c10::kFloat;
    }
}

QUICKMLP_NAMESPACE_END

struct QuickMLPBindings : public torch::CustomClassHolder
{
    static void setCompileDebugMode(bool enable)
    {
        qmlp::QuickMLP::Instance().setCompileDebugMode(enable);
    }
    static bool isCompileDebugMode()
    {
        return qmlp::QuickMLP::Instance().isCompileDebugMode();
    }
    static void setLogLevel(const std::string& level)
    {
        qmlp::QuickMLP::Instance().setLogLevel(level);
    }
    static bool isCudaAvailable()
    {
        return qmlp::QuickMLP::Instance().isCudaAvailable();
    }
};

namespace
{
    bool BindingsInitialized = false;
}

TORCH_LIBRARY(qmlp_cu, m)
{
    m.class_<QuickMLPBindings>("QuickMLP")
        .def_static("set_compile_debug_mode", &QuickMLPBindings::setCompileDebugMode,
            "If set to true, CUDA kernels are compiled in debug mode '-D'")
        .def_static("is_compile_debug_mode", &QuickMLPBindings::isCompileDebugMode,
            "Returns if debug compilation enabled for CUDA kernels or not")
        .def_static("set_log_level", &QuickMLPBindings::setLogLevel,
            "Sets the log level, one of 'off', 'debug', 'info', 'warn', 'error'")
        .def_static("is_cuda_available", &QuickMLPBindings::isCudaAvailable,
            "Returns true iff CUDA is available. If false, all kernel calls will fail with an exception.")
    ;

    bindActivation(m);
    bindEncoding(m);
    bindNetwork(m);
    bindUtils(m);

    qmlp::QuickMLP::Instance().getLogger()->info("QuickMLP library initialized");
    BindingsInitialized = true;
}

QUICKMLP_NAMESPACE_BEGIN

void InitBindings()
{
    //the static initializer in TORCH_LIBRARY should have been called.
    //Check it
    if (!BindingsInitialized)
    {
        qmlp::QuickMLP::Instance().getLogger()->error("QuickMLP bindings not loaded!!");
        throw std::runtime_error("QuickMLP bindings not loaded!!");
    }
}

QUICKMLP_NAMESPACE_END


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)

/**
 * On windows built, PyInit_ ## TORCH_EXTENSION_NAME is somehow always exported.
 * But we do not actually provide it as this is not a direct importable python extension.
 * It is loaded instead through PyTorch.
 * Hence -> define dummy
 */

__declspec(dllexport) void PyInit_qmlp_cu() {}

#endif
