#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

#include "test_against_eigen.h"

using namespace qmlp;
using namespace qmlp::tests;

TEMPLATE_TEST_CASE_SIG("test-agaist-eigen-1", "[eigen]",
    ((int Channels0, int Channels1, TestActivationType Activ1),
        Channels0, Channels1, Activ1),
    //(16, 16, TestActivationType::SINE),
    //(16, 16, TestActivationType::RELU),
    //(16, 16, TestActivationType::IDENTITY),
    (16, 32, TestActivationType::SINE),
    (16, 32, TestActivationType::RELU),
    (16, 32, TestActivationType::IDENTITY),
    (48, 16, TestActivationType::SINE),
    (48, 16, TestActivationType::RELU),
    (48, 16, TestActivationType::IDENTITY),
    (48, 32, TestActivationType::SINE),
    (48, 32, TestActivationType::RELU),
    (48, 32, TestActivationType::IDENTITY)
)
{
    nlohmann::json cfg = {
        {"num_inputs", Channels0},
        {"num_outputs", Channels1},
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
            })
        })}
    };
    std::filesystem::path current(__FILE__);
    auto root = current.parent_path().parent_path();
    auto configFolder = root / "network_configs";

    int N = 32;
    CUstream stream = nullptr;

    //create network
    auto network = std::make_shared<qmlp::FusedNetwork>(cfg, configFolder);
    WARN("network parameters: " << network->networkParameterCount());

    //create parameter tensor
    std::default_random_engine rng(42);
    qmlp::Tensor parametersDevice(
        network->networkParameterPrecision(qmlp::Tensor::INFERENCE),
        { network->networkParameterCount() });
    parametersDevice.zero_();
    network->setNetworkParameter(parametersDevice, qmlp::Tensor::INFERENCE);
    network->initializeInferenceParameters(rng);
    qmlp::Tensor gradientsDevice(
        network->networkParameterPrecision(qmlp::Tensor::GRADIENTS),
        { network->networkParameterCount() });
    network->setNetworkParameter(gradientsDevice, qmlp::Tensor::GRADIENTS);

    //create input and output tensors
    qmlp::Tensor inputDevice(network->precisionIn(), { N, network->channelsIn() });
    qmlp::Tensor outputDevice(network->precisionOut(), { N, network->channelsOut() });
    EigenMatrixX inputHost = EigenMatrixX::Random(N, network->channelsIn());
    toGpuTensor(inputDevice, inputHost);

    //INFERENCE
    //run cuda network
    network->inference(inputDevice, outputDevice, stream);
    CKL_SAFE_CALL(cudaDeviceSynchronize());
    EigenMatrixX outputCudaHost = toEigenMatrix(outputDevice);
    //run Eigen network
    EigenMatrixX outputEigenHost;
    {
        auto input = inputHost.transpose();
        EigenMatrixX weights0 = toEigenMatrix(network->networkParameter(0, false, qmlp::Tensor::INFERENCE));
        EigenVectorX bias0 = toEigenVector(network->networkParameter(0, true, qmlp::Tensor::INFERENCE));
        EigenMatrixX outTemp0 = (weights0 * input).colwise() + bias0;
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);
        outputEigenHost = out0.transpose();
    }
    //compare
    {
        INFO("output CUDA:\n" << outputCudaHost);
        INFO("output Eigen:\n" << outputEigenHost);
        compareEigen(outputCudaHost, outputEigenHost);
    }

    //DERIVATIVES

    //adjoint output
    EigenMatrixX adjOutputHost = EigenMatrixX::Random(N, network->channelsOut());
    qmlp::Tensor adjOutputDevice(network->precisionOut(), { N, network->channelsOut() });
    toGpuTensor(adjOutputDevice, adjOutputHost);

    //first run Eigen
    EigenMatrixX adjInputEigen, adjWeights0Eigen, adjBias0Eigen;
    {
        //forward
        auto input = inputHost.transpose();
        EigenMatrixX weights0 = toEigenMatrix(network->networkParameter(0, false, qmlp::Tensor::INFERENCE));
        EigenVectorX bias0 = toEigenVector(network->networkParameter(0, true, qmlp::Tensor::INFERENCE));
        EigenMatrixX outTemp0 = (weights0 * input).colwise() + bias0;
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);

        //backward
        EigenMatrixX adjOut0 = adjOutputHost.transpose();
        std::cout << "adjOut0 = " << adjOut0.block(0, 0, adjOut0.rows(), 1).transpose() << "\n";

        EigenMatrixX adjOutTemp0 = TestActivation<Activ1>::adjoint(outTemp0, adjOut0);
        std::cout << "adjOutTemp0 = " << adjOutTemp0.block(0, 0, adjOutTemp0.rows(), 1).transpose() << "\n";
        adjBias0Eigen = adjOutTemp0.rowwise().sum();
        adjWeights0Eigen = adjOutTemp0 * input.transpose();
        EigenMatrixX adjInput = weights0.transpose() * adjOutTemp0;
        std::cout << "adjInput = " << adjInput.block(0, 0, adjInput.rows(), 1).transpose() << "\n";
        adjInputEigen = adjInput.transpose();
    }

    //run CUDA in the different configurations
    qmlp::Tensor adjInputDevice(network->precisionIn(), { N, network->channelsIn() });
    const auto adjointWithFlags = [&](int flags)
    {
        return qmlp::tests::adjointWithFlags(network, inputDevice, outputDevice, adjInputDevice, adjOutputDevice, flags, stream);
    };

    //only input derivatives
    {
        int tmpSize = adjointWithFlags(qmlp::FusedNetwork::GRADIENTS_INPUT);
        INFO("size of temporary memory: " << tmpSize);
        COMPARE_TENSOR_AND_MATRIX(adjInputDevice, adjInputEigen);
    }
}

