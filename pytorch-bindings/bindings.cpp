#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include <qmlp/activation.h>
#include <qmlp/qmlp.h>

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <Windows.h>
#endif


namespace py = pybind11;

static QUICKMLP_NAMESPACE::Tensor wrap(const torch::Tensor& t)
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

struct ActivationCache
{
    static QUICKMLP_NAMESPACE::Activation_ptr create(const std::string& cfg)
    {
        typedef std::unordered_map<std::string, QUICKMLP_NAMESPACE::Activation_ptr> Cache_t;
        static Cache_t cache;

        auto it = cache.find(cfg);
        if (it != cache.end()) return it->second;

        static QUICKMLP_NAMESPACE::ActivationFactory factory(QUICKMLP_NAMESPACE::QuickMLP::Instance().kernelLoader());
        auto a = factory.getOrInline(cfg);
        cache.emplace(cfg, a);
        return a;
    }
};

class ActivationBindings : public torch::CustomClassHolder
{
    std::string cfg_;
    QUICKMLP_NAMESPACE::Activation_ptr a_;

public:
    ActivationBindings(const std::string& cfg)
        : cfg_(cfg), a_(ActivationCache::create(cfg))
    {}

    [[nodiscard]] std::string cfg() const
    {
        return cfg_;
    }

    torch::Tensor forward(const torch::Tensor& input)
    {
        torch::Tensor output = torch::empty_like(input);

        auto inputWrapped = wrap(input);
        auto outputWrapped = wrap(output);
        CUstream stream = c10::cuda::getCurrentCUDAStream();
        a_->forward(inputWrapped, outputWrapped, stream);

        return output;
    }

    torch::Tensor adjoint(const torch::Tensor& input, const torch::Tensor& adjOutput)
    {
        torch::Tensor adjInput = torch::empty_like(input);

        auto inputWrapped = wrap(input);
        auto adjOutputWrapped = wrap(adjOutput);
        auto adjInputWrapped = wrap(adjInput);
        CUstream stream = c10::cuda::getCurrentCUDAStream();
        a_->adjoint(inputWrapped, adjOutputWrapped, adjInputWrapped, stream);

        return adjInput;
    }
};


TORCH_LIBRARY(qmlp, m)
{
    m.class_<ActivationBindings>("Activation")
        .def(torch::init<std::string>())
        .def("forward", &ActivationBindings::forward)
        .def("adjoint", &ActivationBindings::adjoint)
        .def_pickle(
            [](const c10::intrusive_ptr<ActivationBindings>& self) -> std::string {
                return self->cfg();
            },
            [](std::string state) -> c10::intrusive_ptr<ActivationBindings> {
                return c10::make_intrusive<ActivationBindings>(std::move(state));
            });
        ;
}
