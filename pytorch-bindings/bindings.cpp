#include <torch/extension.h>
#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <unordered_map>

#include <qmlp/activation.h>

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <Windows.h>
#endif


namespace py = pybind11;

static QUICKMLP_NAMESPACE::Tensor wrap(const torch::Tensor& t)
{
    qmlp::Tensor::Precision p;
    if (t.dtype() == c10::kFloat)
        p = qmlp::Tensor::FLOAT;
    else if (t.dtype() == c10::kHalf)
        p = qmlp::Tensor::HALF;
    else
        throw std::runtime_error("Unsupported datatype, only float and half tensors supported");

}

struct ActivationBindings
{
    typedef std::unordered_map<std::string, QUICKMLP_NAMESPACE::Activation_ptr> ActivationCache_t;
    static ActivationCache_t cache;

    static void clear_cache()
    {
        cache.clear();
    }

    static int64_t dummy(int64_t v)
    {
        return v + 1;
    }

    //static std::string get_activation(const std::string& cfg)
    //{
    //    
    //}
};
ActivationBindings::ActivationCache_t ActivationBindings::cache;

static void TORCH_LIBRARY_init_qmlp(torch::Library&);
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_qmlp(
    torch::Library::DEF, &TORCH_LIBRARY_init_qmlp, "qmlp", c10::nullopt, __FILE__, __LINE__);

void TORCH_LIBRARY_init_qmlp(torch::Library& m)
{
    m.def("activation_clear_cache", ActivationBindings::clear_cache);
    m.def("dummy", ActivationBindings::dummy);
}
