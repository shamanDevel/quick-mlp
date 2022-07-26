#include "bind_encoding.h"

#include <torch/extension.h>
#include <torch/types.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

static torch::Tensor createOutputTensor(const torch::Tensor& input, int channels)
{
    TORCH_CHECK(input.dim() == 2, "Inputs must be 2D (B,C)");
    auto B = input.size(0);
    auto t = torch::empty({ B, channels }, at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA).memory_format(c10::MemoryFormat::Contiguous));
    TORCH_CHECK(t.stride(1) == 1);
    return t;
}

std::string EncodingBindings::toJson() const
{
    return a_->toJson().dump();
}

torch::Tensor EncodingBindings::inference(const torch::Tensor& input) const
{
    TORCH_CHECK(!hasParameters(), "Encoding has parameters, call the variation of 'inference' with parameters");
    torch::Tensor output = createOutputTensor(input, a_->numOutputChannels());

    auto inputWrapped = QUICKMLP_NAMESPACE::wrap(input);
    auto outputWrapped = QUICKMLP_NAMESPACE::wrap(output);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    a_->forward(inputWrapped, outputWrapped, stream, {});

    return output;
}

torch::Tensor EncodingBindings::inferenceWithParameter(const torch::Tensor& input,
    const torch::Tensor& parameterForward) const
{
    TORCH_CHECK(hasParameters(), "Encoding doesn't have parameters, call the variation of 'inference' without parameters");
    torch::Tensor output = createOutputTensor(input, a_->numOutputChannels());

    auto inputWrapped = QUICKMLP_NAMESPACE::wrap(input);
    auto paramWrapped = QUICKMLP_NAMESPACE::wrap(parameterForward);
    auto outputWrapped = QUICKMLP_NAMESPACE::wrap(output);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    a_->forward(inputWrapped, outputWrapped, stream, paramWrapped);

    return output;
}

torch::Tensor EncodingBindings::adjoint(const torch::Tensor& input, const torch::Tensor& adjOutput) const
{
    TORCH_CHECK(!hasParameters(), "Encoding has parameters, call the variation of 'adjoint' with parameters");
    torch::Tensor adjInput = torch::zeros_like(input);

    auto inputWrapped = QUICKMLP_NAMESPACE::wrap(input);
    auto adjOutputWrapped = QUICKMLP_NAMESPACE::wrap(adjOutput);
    auto adjInputWrapped = QUICKMLP_NAMESPACE::wrap(adjInput);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    a_->adjoint(inputWrapped, adjOutputWrapped, adjInputWrapped, stream, 
        {}, {}, qmlp::IEncoding::ALL_GRADIENTS);

    return adjInput;
}

std::tuple<torch::Tensor, torch::Tensor> EncodingBindings::adjointWithParameter(const torch::Tensor& input,
    const torch::Tensor& parameterForward, const torch::Tensor& adjOutput) const
{
    return adjointWithParameterAndFlags(input, parameterForward, adjOutput, qmlp::IEncoding::ALL_GRADIENTS);
}
std::tuple<torch::Tensor, torch::Tensor> EncodingBindings::adjointWithParameterAndFlags(const torch::Tensor & input,
    const torch::Tensor & parameterForward, const torch::Tensor & adjOutput, int flags) const
{
    TORCH_CHECK(hasParameters(), "Encoding doesn't have parameters, call the variation of 'adjoint' without parameters");
    torch::Tensor adjInput = torch::zeros_like(input);
    torch::Tensor adjParam = torch::zeros_like(parameterForward);

    auto inputWrapped = QUICKMLP_NAMESPACE::wrap(input);
    auto paramWrapped = QUICKMLP_NAMESPACE::wrap(parameterForward);
    auto adjOutputWrapped = QUICKMLP_NAMESPACE::wrap(adjOutput);
    auto adjInputWrapped = QUICKMLP_NAMESPACE::wrap(adjInput);
    auto adjParamWrapped = QUICKMLP_NAMESPACE::wrap(adjParam);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    a_->adjoint(inputWrapped, adjOutputWrapped, adjInputWrapped, stream,
        paramWrapped, adjParamWrapped, flags);

    return { adjInput, adjParam };
}

torch::Tensor EncodingBindings::forward(c10::intrusive_ptr<EncodingBindings> self, const torch::Tensor& input)
{
    TORCH_CHECK(!self->hasParameters(), "Encoding has parameters, call the variation of 'forward' with parameters");
    torch::autograd::tensor_list ret = EncodingAutogradFunction::apply(input, self);
    return ret[0];
}

torch::Tensor EncodingBindings::forwardWithParameter(c10::intrusive_ptr<EncodingBindings> self,
    const torch::Tensor& input, const torch::Tensor& parameter)
{
    TORCH_CHECK(self->hasParameters(), "Encoding doesn't have parameters, call the variation of 'forward' without parameters");
    torch::autograd::tensor_list ret = EncodingAutogradFunctionWithParameter::apply(input, parameter, self);
    return ret[0];
}

torch::autograd::variable_list EncodingAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
                                                                   const torch::Tensor& input, EncodingBindings_ptr activ)
{
    ctx->save_for_backward({ input });
    ctx->saved_data["activ"] = activ;

    torch::Tensor output = activ->inference(input);
    return { output };
}

torch::autograd::tensor_list EncodingAutogradFunction::backward(torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    torch::Tensor input = saved[0];
    EncodingBindings_ptr activ = ctx->saved_data["activ"].toCustomClass<EncodingBindings>();

    torch::Tensor gradOutput = grad_outputs[0];
    torch::Tensor gradInput = activ->adjoint(input, gradOutput);
    return { gradInput, torch::Tensor() };
}

torch::autograd::variable_list EncodingAutogradFunctionWithParameter::forward(
    torch::autograd::AutogradContext* ctx,
    const torch::Tensor& input, const torch::Tensor& parameter, EncodingBindings_ptr activ)
{
    ctx->save_for_backward({ input, parameter });
    ctx->saved_data["activ"] = activ;
    ctx->saved_data["hasInputGradients"] = input.requires_grad();
    ctx->saved_data["hasParamGradients"] = parameter.requires_grad();

    torch::Tensor output = activ->inferenceWithParameter(input, parameter);
    return { output };
}

torch::autograd::tensor_list EncodingAutogradFunctionWithParameter::backward(torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    torch::Tensor input = saved[0];
    torch::Tensor parameter = saved[1];
    EncodingBindings_ptr activ = ctx->saved_data["activ"].toCustomClass<EncodingBindings>();
    bool hasInputGradients = ctx->saved_data["hasInputGradients"].toBool();
    bool hasParamGradients = ctx->saved_data["hasParamGradients"].toBool();
    int flags =
        (hasInputGradients ? QUICKMLP_NAMESPACE::IEncoding::INPUT_GRADIENTS : 0) |
        (hasParamGradients ? QUICKMLP_NAMESPACE::IEncoding::PARAM_GRADIENTS : 0);

    torch::Tensor gradOutput = grad_outputs[0];
    auto grad = activ->adjointWithParameterAndFlags(input, parameter, gradOutput, flags);
    return { std::get<0>(grad), std::get<1>(grad), torch::Tensor() };
}

void bindEncoding(torch::Library& m)
{
    m.class_<EncodingBindings>("Encoding")
        .def(torch::init<std::string>())
        .def("max_input_channel", &EncodingBindings::maxInputChannel)
        .def("num_output_channels", &EncodingBindings::numOutputChannels)
        .def("has_parameters", &EncodingBindings::hasParameters)
        .def("to_json", &EncodingBindings::toJson)
        .def("create_inference_parameters", &EncodingBindings::createInferenceParameters)
        .def("create_gradient_parameters", &EncodingBindings::createGradientParameters)
        .def("inference", &EncodingBindings::inference)
        .def("adjoint", &EncodingBindings::adjoint)
        .def("inference_with_parameter", &EncodingBindings::inferenceWithParameter)
        .def("adjoint_with_parameter", &EncodingBindings::adjointWithParameter)
        .def("forward", [](const EncodingBindings_ptr& self, const torch::Tensor& input)
        {
            //detour over a static function to get the intrusive pointer
            return EncodingBindings::forward(self, input);
        })
        .def("forward_with_parameter", [](const EncodingBindings_ptr& self,
                const torch::Tensor& input, const torch::Tensor& parameter)
            {
                //detour over a static function to get the intrusive pointer
                return EncodingBindings::forwardWithParameter(self, input, parameter);
            })
        .def_pickle(
            [](const c10::intrusive_ptr<EncodingBindings>& self) -> std::string {
                return self->cfg();
            },
            [](std::string state) -> c10::intrusive_ptr<EncodingBindings> {
                return c10::make_intrusive<EncodingBindings>(std::move(state));
            })
    ;
}
