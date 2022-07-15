#include <qmlp/activation.h>

#include <ckl/errors.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <qmlp/errors.h>

#include "qmlp/qmlp.h"
#include "qmlp/kernels/tensor.cuh"

QUICKMLP_NAMESPACE_BEGIN
    static std::string generateCode(const std::string& id, const std::string& forward, const std::string& adjoint)
{
    static const char* TEMPLATE = R"code(
struct %s
{
    static __device__ half forward(half x) {
        half z;
        %s ;
        return z;
    }
    static __device__ half adjoint(half x, half adjz) {
        half adjx;
        %s ;
        return adjx;
    }
};
    )code";

    //https://stackoverflow.com/a/3588492/1786598
    static const std::regex CPP_IDENTIFIER_REGEX("^[a-zA-Z_][a-zA-Z0-9_]*$");

    if (!std::regex_match(id, CPP_IDENTIFIER_REGEX))
    {
        throw configuration_error("id of the activation function is not a valid C++ identifier: %s", id.c_str());
    }

    return ckl::internal::Format::format(TEMPLATE, id.c_str(), forward.c_str(), adjoint.c_str());
}

Activation::Activation(const std::string& id, const std::string& forward, const std::string& adjoint)
    : id_(id)
    , forward_(forward)
    , adjoint_(adjoint)
    , code_(generateCode(id, forward, adjoint))
{
    
}

Activation::Activation(const nlohmann::json& cfg)
    : Activation(cfg.at("id"), cfg.at("forward"), cfg.at("adjoint"))
{}

nlohmann::json Activation::toJson() const
{
    return nlohmann::json{
        {"id", id()},
        {"forward", forward()},
        {"adjoint", adjoint()}
    };
}

void Activation::forward(const Tensor& input, Tensor& output, CUstream stream)
{
    //Check shapes
    CHECK_DIM(input, 2);
    CHECK_DIM(output, 2);
    CHECK_DTYPE(input, Tensor::HALF);
    CHECK_DTYPE(output, Tensor::HALF);
    CHECK_SIZE(output, 0, input.size(0));
    CHECK_SIZE(output, 1, input.size(1));
    int numel = input.size(0) * input.size(1);

    if (!forwardKernel_.has_value()) {
        //generate code
        auto kl = QuickMLP::Instance().kernelLoader();
        std::string codeTemplate = kl->findFile("qmlp/kernels/activation_kernels.cuh").value();
        replaceAll(codeTemplate, "$$DEFINE_ACTIVATIONS$$", code());
        replaceAll(codeTemplate, "$$ACTIVATION_ID$$", id());

        int compileFlags = ckl::KernelLoader::CompilationFlags::CompileThrowOnError;
#ifndef NDEBUG
        compileFlags |= ckl::KernelLoader::CompilationFlags::CompileDebugMode
            | ckl::KernelLoader::CompilationFlags::CompileVerboseLogging;
#endif
        ckl::KernelFunction fun = kl->getKernel(
            "qmlp::kernel::ActivationForwardKernel",
            codeTemplate,
            {},
            compileFlags).value();

        forwardKernel_ = fun;
    }
    auto& fun = forwardKernel_.value();

    //launch kernel
    int minGridSize = std::min(
        CKL_DIV_UP(numel, fun.bestBlockSize()),
        fun.minGridSize());
    dim3 virtual_size(input.size(0), input.size(1));
    auto inputAcc = input.accessor<kernel::Tensor2Read<half>>();
    auto outputAcc = output.accessor<kernel::Tensor2RW<half>>();
    fun.call(
        minGridSize, fun.bestBlockSize(), 0, stream,
        virtual_size, inputAcc, outputAcc);
}

void Activation::adjoint(const Tensor& input, const Tensor& adjOutput, Tensor& adjInput, CUstream stream)
{
    //Check shapes
    CHECK_DIM(input, 2);
    CHECK_DIM(adjOutput, 2);
    CHECK_DIM(adjInput, 2);
    CHECK_DTYPE(input, Tensor::HALF);
    CHECK_DTYPE(adjOutput, Tensor::HALF);
    CHECK_DTYPE(adjInput, Tensor::HALF);
    CHECK_SIZE(adjOutput, 0, input.size(0));
    CHECK_SIZE(adjOutput, 1, input.size(1));
    CHECK_SIZE(adjInput, 0, input.size(0));
    CHECK_SIZE(adjInput, 1, input.size(1));
    int numel = input.size(0) * input.size(1);

    if (!adjointKernel_.has_value())
    {
        //generate code
        auto kl = QuickMLP::Instance().kernelLoader();
        std::string codeTemplate = kl->findFile("qmlp/kernels/activation_kernels.cuh").value();
        replaceAll(codeTemplate, "$$DEFINE_ACTIVATIONS$$", code());
        replaceAll(codeTemplate, "$$ACTIVATION_ID$$", id());

        int compileFlags = ckl::KernelLoader::CompilationFlags::CompileThrowOnError;
#ifndef NDEBUG
        compileFlags |= ckl::KernelLoader::CompilationFlags::CompileDebugMode
            | ckl::KernelLoader::CompilationFlags::CompileVerboseLogging;
#endif
        ckl::KernelFunction fun = kl->getKernel(
            "qmlp::kernel::ActivationAdjointKernel",
            codeTemplate,
            {},
            compileFlags).value();

        adjointKernel_ = fun;
    }
    auto& fun = adjointKernel_.value();

    //launch kernel
    int minGridSize = std::min(
        CKL_DIV_UP(numel, fun.bestBlockSize()),
        fun.minGridSize());
    dim3 virtual_size(input.size(0), input.size(1));
    auto inputAcc = input.accessor<kernel::Tensor2Read<half>>();
    auto adjOutputAcc = adjOutput.accessor<kernel::Tensor2Read<half>>();
    auto adjInputAcc = adjInput.accessor<kernel::Tensor2RW<half>>();
    fun.call(
        minGridSize, fun.bestBlockSize(), 0, stream,
        virtual_size, inputAcc, adjOutputAcc, adjInputAcc);
}

ActivationFactory::ActivationFactory(const nlohmann::json& cfg, const std::filesystem::path& parent,
                                     ckl::KernelLoader_ptr loader)
{
    if (!cfg.is_array())
    {
        throw configuration_error("The activation specification tag must contain an array");
    }

    for (const auto& element : cfg)
    {
        if (element.type() == nlohmann::detail::value_t::string)
        {
            std::string filename = element;
            std::filesystem::path path;
            //search for that file
            //1. Direct
            path = filename;
            if (std::filesystem::exists(path))
            {
                parseFile(path);
                continue;
            }
            if (path.is_absolute())
            {
                throw configuration_error("Absolute path '%s' does not exist. Can't continue resolving the activation file, relative path required",
                    filename.c_str());
            }
            //2. Relative to parent
            path = parent / filename;
            if (std::filesystem::exists(path))
            {
                parseFile(path);
                continue;
            }
            //3. built-in activations
            auto content = loader->findFile(filename);
            if (content.has_value())
            {
                auto j = nlohmann::json::parse(content.value());
                parseFile(j);
                continue;
            }
            //not found
            throw configuration_error("Path to activation file '%s' could not be resolved. Is the filename correct?",
                filename.c_str());
        }
        else if (element.type() == nlohmann::detail::value_t::object)
        {
            //inline specification
            parseActivation(element, true);
        }
        else
        {
            std::string es = element.dump();
            throw configuration_error("Unknown type of element '%s', only filenames (strings) or in-place specifications (objects) supported.",
                es.c_str());
        }
    }
}

ActivationFactory::ActivationFactory(ckl::KernelLoader_ptr loader)
{
    auto content = loader->findFile("qmlp/builtin-activations.json");
    if (content.has_value())
    {
        auto j = nlohmann::json::parse(content.value());
        parseFile(j);
    } else
    {
        throw configuration_error("File with builtin activations not found");
    }
}

Activation_ptr ActivationFactory::get(const std::string& key) const
{
    auto it = activations_.find(key);
    if (it == activations_.end())
        throw configuration_error("No activation with id '%s' found!", key.c_str());
    return it->second;
}

Activation_ptr ActivationFactory::getOrInline(const std::string& key)
{
    if (key.empty()) throw std::runtime_error("empty key");
    if (key[0] == '{')
    {
        //inline activation
        nlohmann::json j = nlohmann::json::parse(key);
        return parseActivation(j, false);
    }
    else
    {
        //default activation
        return get(key);
    }
}

void ActivationFactory::parseFile(const std::filesystem::path& file)
{
    nlohmann::json j;
    {
        std::ifstream i(file);
        i >> j;
    }
    if (!j.is_array())
    {
        throw configuration_error("Activation file '%s' is expected to contain an array object as the root element",
            file.c_str());
    }

    parseFile(j);
}

void ActivationFactory::parseFile(const nlohmann::json& j)
{
    assert(j.is_array());
    //loop over the array
    for (const auto& element : j)
    {
        parseActivation(element, true);
    }
}

Activation_ptr ActivationFactory::parseActivation(const nlohmann::json& cfg, bool emplace)
{
    try
    {
        Activation_ptr a = std::make_shared<Activation>(cfg);
        if (emplace) {
            activations_.emplace(a->id(), a);
        }
        return a;
    } catch (const std::exception& ex)
    {
        std::throw_with_nested(configuration_error("Parsing the activation configuration failed."));
    }
}

QUICKMLP_NAMESPACE_END
