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
#include "bind_utils.h"

QUICKMLP_NAMESPACE_BEGIN
QUICKMLP_NAMESPACE::Tensor wrap(const torch::Tensor& t)
{
    if (!t.is_cuda())
        throw std::runtime_error("Tensor must reside on CUDA");

    qmlp::Tensor::Precision p;
    if (t.dtype() == c10::kFloat)
        p = qmlp::Tensor::FLOAT;
    else if (t.dtype() == c10::kHalf)
        p = qmlp::Tensor::HALF;
    else
        throw std::runtime_error("Unsupported datatype, only float and half tensors supported");

    std::vector<int32_t> sizes(t.sizes().begin(), t.sizes().end());
    std::vector<int32_t> strides(t.strides().begin(), t.strides().end());

    return QUICKMLP_NAMESPACE::Tensor( t.data_ptr(), p, sizes, strides );
}
QUICKMLP_NAMESPACE_END

struct QuickMLPBindings : public torch::CustomClassHolder
{
    static void setDebugMode(bool enable)
    {
        qmlp::QuickMLP::Instance().setDebugMode(enable);
    }
};

namespace
{
    bool BindingsInitialized = false;
}

TORCH_LIBRARY(qmlp, m)
{
    m.class_<QuickMLPBindings>("QuickMLP")
        .def_static("set_debug_mode", &QuickMLPBindings::setDebugMode)
    ;

    bindActivation(m);
    bindEncoding(m);
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
