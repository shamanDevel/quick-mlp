#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

enum class TestActivationType
{
    RELU, SINE
};
//name for the config file
static const char* TestActivationConfigName[] = {
    "relu", "sine"
};
template<TestActivationType T> struct TestActivation;
template<>
struct TestActivation<TestActivationType::RELU>
{
    static Eigen::MatrixXf forward(const Eigen::MatrixXf& x)
    {
        return x.cwiseMin(0.0f);
    }
    static Eigen::MatrixXf adjoint(const Eigen::MatrixXf& x, const Eigen::MatrixXf& adjz)
    {
        return ((x.array() > 0.f).cast<float>() * adjz.array()).matrix();
    }
};
template<>
struct TestActivation<TestActivationType::SINE>
{
    static Eigen::MatrixXf forward(const Eigen::MatrixXf& x)
    {
        return x.array().sin().matrix();
    }
    static Eigen::MatrixXf adjoint(const Eigen::MatrixXf& x, const Eigen::MatrixXf& adjz)
    {
        return (x.array().cos() * adjz.array()).matrix();
    }
};


TEMPLATE_TEST_CASE_SIG("test-agaist-eigen", "[eigen]", 
    ((int Channels0, int Channels1, TestActivationType Activ1, int Channels2, TestActivationType Activ2),
        Channels0, Channels1, Activ1, Channels2, Activ2),
    (16, 16, TestActivationType::SINE, 16, TestActivationType::SINE),
    (16, 16, TestActivationType::RELU, 16, TestActivationType::RELU),
    (16, 16, TestActivationType::SINE, 32, TestActivationType::SINE),
    (16, 16, TestActivationType::RELU, 32, TestActivationType::RELU),
    (32, 48, TestActivationType::SINE, 16, TestActivationType::SINE),
    (32, 48, TestActivationType::RELU, 16, TestActivationType::RELU),
    (32, 48, TestActivationType::SINE, 32, TestActivationType::SINE),
    (32, 48, TestActivationType::RELU, 32, TestActivationType::RELU))
{
    nlohmann::json cfg = {
        {"num_inputs", Channels0},
        {"num_outputs", Channels2},
        {"activation_specification", nlohmann::json::array({
            "qmlp/builtin-activations.json"
        }) },
        {"encodings", nlohmann::json::array({
            nlohmann::json::object({
                {"id", "identity"},
                {"start_in", 0},
                {"n_in", Channels0}
            })
        })} ,
        {"network", nlohmann::json::array({
            nlohmann::json::object({
                {"n_out", Channels1},
                {"bias", true},
                {"activation", TestActivationConfigName[int(Activ1)]}
            }),
            nlohmann::json::object({
                {"n_out", Channels2},
                {"bias", true},
                {"activation", TestActivationConfigName[int(Activ2)]}
            }),
        })}
    };
    std::filesystem::path current(__FILE__);
    auto root = current.parent_path().parent_path();
    auto configFolder = root / "network_configs";

    //create network
    auto network = std::make_shared<qmlp::FusedNetwork>(cfg, configFolder);
    WARN("network parameters: " << network->networkParameterCount());
}

