#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

#include "test_against_eigen.h"

using namespace qmlp;
using namespace qmlp::tests;

TEMPLATE_TEST_CASE_SIG("test-against-eigen-2", "[eigen]", 
    ((int Channels0, int Channels1, bool Bias1, TestActivationType Activ1, int Channels2, bool Bias2, TestActivationType Activ2),
        Channels0, Channels1, Bias1, Activ1, Channels2, Bias2, Activ2),
    (16, 16, false, TestActivationType::SINE, 16, false, TestActivationType::IDENTITY)
    //(16, 16, false, TestActivationType::SINE, 16, false, TestActivationType::SINE),
    //(16, 16, false, TestActivationType::RELU, 16, false, TestActivationType::IDENTITY),
    //(16, 16, false, TestActivationType::SINE, 32, false, TestActivationType::SINE),
    //(16, 16, false, TestActivationType::RELU, 32, false, TestActivationType::IDENTITY),
    //(32, 48, false, TestActivationType::SINE, 16, false, TestActivationType::SINE),
    //(32, 48, false, TestActivationType::RELU, 16, false, TestActivationType::IDENTITY),
    //(32, 48, false, TestActivationType::SINE, 32, false, TestActivationType::SINE),
    //(32, 48, false, TestActivationType::RELU, 32, false, TestActivationType::IDENTITY),
    //(16, 16, true, TestActivationType::SINE, 16, true, TestActivationType::IDENTITY),
    //(16, 16, true, TestActivationType::SINE, 16, true, TestActivationType::SINE),
    //(16, 16, true, TestActivationType::RELU, 16, true, TestActivationType::IDENTITY),
    //(16, 16, true, TestActivationType::SINE, 32, true, TestActivationType::SINE),
    //(16, 16, true, TestActivationType::RELU, 32, true, TestActivationType::IDENTITY),
    //(32, 48, true, TestActivationType::SINE, 16, true, TestActivationType::SINE),
    //(32, 48, true, TestActivationType::RELU, 16, true, TestActivationType::IDENTITY),
    //(32, 48, true, TestActivationType::SINE, 32, true, TestActivationType::SINE),
    //(32, 48, true, TestActivationType::RELU, 32, true, TestActivationType::IDENTITY)
    )
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
                {"bias", Bias1},
                {"activation", TestActivationConfigName[int(Activ1)]}
            }),
            nlohmann::json::object({
                {"n_out", Channels2},
                {"bias", Bias2},
                {"activation", TestActivationConfigName[int(Activ2)]}
            }),
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
        EigenMatrixX weights1 = toEigenMatrix(network->networkParameter(1, false, qmlp::Tensor::INFERENCE));
        EigenVectorX bias1 = toEigenVector(network->networkParameter(1, true, qmlp::Tensor::INFERENCE));
        EigenMatrixX outTemp0 = Bias1
            ? ((weights0 * input).colwise() + bias0).eval()
            : (weights0 * input).eval();
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);
        EigenMatrixX outTemp1 = Bias2
            ? ((weights1 * out0).colwise() + bias1).eval()
            : (weights1 * out0).eval();
        EigenMatrixX out1 = TestActivation<Activ2>::forward(outTemp1);
        outputEigenHost = out1.transpose();
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
    EigenMatrixX adjInputEigen, adjWeights0Eigen, adjBias0Eigen, adjWeights1Eigen, adjBias1Eigen;
    {
        //forward
        auto input = inputHost.transpose();
        EigenMatrixX weights0 = toEigenMatrix(network->networkParameter(0, false, qmlp::Tensor::INFERENCE));
        EigenVectorX bias0 = toEigenVector(network->networkParameter(0, true, qmlp::Tensor::INFERENCE));
        EigenMatrixX weights1 = toEigenMatrix(network->networkParameter(1, false, qmlp::Tensor::INFERENCE));
        EigenVectorX bias1 = toEigenVector(network->networkParameter(1, true, qmlp::Tensor::INFERENCE));
        EigenMatrixX outTemp0 = Bias1
            ? ((weights0 * input).colwise() + bias0).eval()
            : (weights0 * input).eval();
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);
        EigenMatrixX outTemp1 = Bias2
            ? ((weights1 * out0).colwise() + bias1).eval()
            : (weights1 * out0).eval();
        EigenMatrixX out1 = TestActivation<Activ2>::forward(outTemp1);
        //backward
        EigenMatrixX adjOut1 = adjOutputHost.transpose();
        std::cout << "adjOut1 = " << adjOut1.block(0, 0, adjOut1.rows(), 1).transpose() << "\n";

        EigenMatrixX adjOutTemp1 = TestActivation<Activ2>::adjoint(outTemp1, adjOut1);
        std::cout << "adjOutTemp1 = " << adjOutTemp1.block(0, 0, adjOutTemp1.rows(), 1).transpose() << "\n";
        adjBias1Eigen = adjOutTemp1.rowwise().sum();
        adjWeights1Eigen = adjOutTemp1 * out0.transpose();
        EigenMatrixX adjOut0 = weights1.transpose() * adjOutTemp1;
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

    //only weight derivatives
    {
        int tmpSize = adjointWithFlags(qmlp::FusedNetwork::GRADIENTS_NETWORK_WEIGHTS);
        INFO("size of temporary memory: " << tmpSize);
        //COMPARE_TENSOR_AND_MATRIX(
        //    network->networkParameter(0, false, Tensor::GRADIENTS),
        //    adjWeights0Eigen);
        COMPARE_TENSOR_AND_MATRIX(
            network->networkParameter(1, false, Tensor::GRADIENTS),
            adjWeights1Eigen);
        if constexpr(Bias1)
        {
            COMPARE_TENSOR_AND_VECTOR(
                network->networkParameter(0, true, Tensor::GRADIENTS),
                adjBias0Eigen);
        }
        if constexpr (Bias2)
        {
            COMPARE_TENSOR_AND_VECTOR(
                network->networkParameter(1, true, Tensor::GRADIENTS),
                adjBias1Eigen);
        }
    }
}

