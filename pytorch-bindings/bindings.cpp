#include "bindings.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include <qmlp/activation.h>
#include <qmlp/qmlp.h>

#include "bind_activation.h"
#include "bind_encoding.h"

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

TORCH_LIBRARY(qmlp, m)
{
    bindActivation(m);
    bindEncoding(m);
}
