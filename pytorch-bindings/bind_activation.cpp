#include "bind_activation.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

#include <qmlp/activation.h>
#include <qmlp/qmlp.h>

QUICKMLP_NAMESPACE::Activation_ptr ActivationCache::create(const std::string& cfg)
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

torch::Tensor ActivationBindings::inference(const torch::Tensor& input) const
{
    torch::Tensor output = torch::empty_like(input);

    auto inputWrapped = wrap(input);
    auto outputWrapped = wrap(output);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    a_->forward(inputWrapped, outputWrapped, stream);

    return output;
}

torch::Tensor ActivationBindings::adjoint(const torch::Tensor& input, const torch::Tensor& adjOutput) const
{
    torch::Tensor adjInput = torch::empty_like(input);

    auto inputWrapped = wrap(input);
    auto adjOutputWrapped = wrap(adjOutput);
    auto adjInputWrapped = wrap(adjInput);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    a_->adjoint(inputWrapped, adjOutputWrapped, adjInputWrapped, stream);

    return adjInput;
}

torch::Tensor ActivationBindings::forward(ActivationBindings_ptr self, const torch::Tensor& input)
{
    torch::autograd::tensor_list ret = ActivationAutogradFunction::apply(input, self);
    return ret[0];
}

torch::autograd::variable_list ActivationAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
    const torch::Tensor& input, ActivationBindings_ptr activ)
{
    ctx->save_for_backward({ input });
    ctx->saved_data["activ"] = activ;

    torch::Tensor output = activ->inference(input);
    return { output };
}

torch::autograd::tensor_list ActivationAutogradFunction::backward(torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    torch::Tensor input = saved[0];
    ActivationBindings_ptr activ = ctx->saved_data["activ"].toCustomClass<ActivationBindings>();

    torch::Tensor gradOutput = grad_outputs[0];
    torch::Tensor gradInput = activ->adjoint(input, gradOutput);
    return { gradInput, torch::Tensor() };
}

void bindActivation(torch::Library& m)
{
    m.class_<ActivationBindings>("Activation")
        .def(torch::init<std::string>())
        .def("inference", &ActivationBindings::forward)
        .def("adjoint", &ActivationBindings::adjoint)
        .def("forward", [](const ActivationBindings_ptr& self, const torch::Tensor& input)
        {
            //detour over a static function to get the intrusive pointer
            return ActivationBindings::forward(self, input);
        })
        .def_pickle(
            [](const c10::intrusive_ptr<ActivationBindings>& self) -> std::string {
                return self->cfg();
            },
            [](std::string state) -> c10::intrusive_ptr<ActivationBindings> {
                return c10::make_intrusive<ActivationBindings>(std::move(state));
            })
    ;
}
