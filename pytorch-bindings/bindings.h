#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <qmlp/qmlp.h>
#include <qmlp/tensor.h>

QUICKMLP_NAMESPACE::Tensor wrap(const torch::Tensor& t);