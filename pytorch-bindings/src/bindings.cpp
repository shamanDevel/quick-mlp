#include "../qmlp/bindings.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include <qmlp/activation.h>
#include <qmlp/qmlp.h>

#include "bind_activation.h"
#include "bind_encoding.h"
#include "bind_network.h"
#include "bind_utils.h"

QUICKMLP_NAMESPACE_BEGIN

QUICKMLP_NAMESPACE::Tensor wrap(const torch::Tensor& t)
{
    TORCH_CHECK(t.is_cuda(), "Tensor must reside on the GPU");

    qmlp::Tensor::Precision p;
    if (t.dtype() == c10::kFloat)
        p = qmlp::Tensor::FLOAT;
    else if (t.dtype() == c10::kHalf)
        p = qmlp::Tensor::HALF;
    else
    {
        TORCH_CHECK(false, "Unsupported datatype, only float and half tensors supported");
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
    static void setVerboseLogging(bool enable)
    {
        qmlp::QuickMLP::Instance().setVerboseLogging(enable);
    }
    static bool isVerboseLogging()
    {
        return qmlp::QuickMLP::Instance().isVerboseLogging();
    }
};

namespace
{
    bool BindingsInitialized = false;
}

TORCH_LIBRARY(qmlp, m)
{
    m.class_<QuickMLPBindings>("QuickMLP")
        .def_static("set_compile_debug_mode", &QuickMLPBindings::setCompileDebugMode,
            "If set to true, CUDA kernels are compiled in debug mode '-D'")
        .def_static("is_compile_debug_mode", &QuickMLPBindings::isCompileDebugMode,
            "Returns if debug compilation enabled for CUDA kernels or not")
        .def_static("set_verbose_logging", &QuickMLPBindings::setVerboseLogging,
            "If set to true, verbose logging messages are enabled")
        .def_static("is_verbose_logging", &QuickMLPBindings::isVerboseLogging,
            "Returns if debug logging is enabled")
    ;

    bindActivation(m);
    bindEncoding(m);
    bindNetwork(m);
    bindUtils(m);

    std::cout << "QuickMLP bindings loaded" << std::endl;
    BindingsInitialized = true;
}

QUICKMLP_NAMESPACE_BEGIN

void InitBindings()
{
    //the static initializer in TORCH_LIBRARY should have been called.
    //Check it
    if (!BindingsInitialized)
    {
        std::cerr << "QuickMLP bindings not loaded!!" << std::endl;
        throw std::runtime_error("QuickMLP bindings not loaded!!");
    }
}

QUICKMLP_NAMESPACE_END
