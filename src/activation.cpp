#include <qmlp/activation.h>

#include <ckl/errors.h>
#include <regex>
#include <qmlp/errors.h>

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
}
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


QUICKMLP_NAMESPACE_END
