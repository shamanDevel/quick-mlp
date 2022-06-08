#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <qmlp/fused_network.h>

TEST_CASE("compilation-test", "[simple]")
{
    std::filesystem::path current(__FILE__);
    auto root = current.parent_path().parent_path();
    auto configFolder = root / "network_configs";
    auto configFile = configFolder / "test1.json";
    nlohmann::json cfg;
    {
        std::ifstream s(configFile);
        REQUIRE(s.is_open());
        s >> cfg;
    }

    int N = 4267;
    std::default_random_engine rng;

    //create network
    auto network = std::make_shared<qmlp::FusedNetwork>(cfg, configFolder);
    WARN("network parameters: " << network->networkParameterCount());

    //create parameter tensor
    qmlp::Tensor parameters(
        network->networkParameterPrecision(qmlp::Tensor::INFERENCE),
        { network->networkParameterCount() });
    parameters.zero_();
    network->setNetworkParameter(parameters, qmlp::Tensor::INFERENCE);
    network->initializeInferenceParameters(rng);

    //create input and output tensors
    qmlp::Tensor input( network->precisionIn(), { N, network->channelsIn() });
    qmlp::Tensor output(network->precisionOut(), { N, network->channelsOut() });
    input.zero_();

    //run network
    CUstream stream = nullptr;
    network->inference(input, output, stream);
    REQUIRE_NOTHROW(CKL_SAFE_CALL(cudaDeviceSynchronize()));
}