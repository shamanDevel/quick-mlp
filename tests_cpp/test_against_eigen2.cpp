#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <spdlog/spdlog.h>

#include <qmlp/fused_network.h>
#include <qmlp/qmlp.h>

#include "test_against_eigen.h"

using namespace qmlp;
using namespace qmlp::tests;

TEMPLATE_TEST_CASE_SIG("test-against-eigen-2", "[eigen]", 
    ((int Channels0, int Channels1, bool Bias1, TestActivationType Activ1, int Channels2, bool Bias2, TestActivationType Activ2),
        Channels0, Channels1, Bias1, Activ1, Channels2, Bias2, Activ2),

    (16, 16, false, TestActivationType::IDENTITY, 16, false, TestActivationType::IDENTITY),
    //(16, 16, false, TestActivationType::SINE,     16, false, TestActivationType::IDENTITY),
    //(16, 16, false, TestActivationType::SINE,     16, false, TestActivationType::SINE),
    //(16, 16, false, TestActivationType::CELU,     16, false, TestActivationType::CELU),

    (19, 16, false, TestActivationType::IDENTITY, 16, false, TestActivationType::IDENTITY),
    (16, 16, false, TestActivationType::IDENTITY, 23, false, TestActivationType::IDENTITY),
    (20, 16, false, TestActivationType::IDENTITY, 24, false, TestActivationType::IDENTITY)//,

    //(16, 16, false, TestActivationType::IDENTITY, 32, false, TestActivationType::IDENTITY),
    //(16, 16, false, TestActivationType::SINE,     32, false, TestActivationType::IDENTITY),
    //(16, 16, false, TestActivationType::SINE,     32, false, TestActivationType::SINE),
    //(16, 16, false, TestActivationType::CELU,     32, false, TestActivationType::CELU),

    //(32, 48, false, TestActivationType::IDENTITY, 16, false, TestActivationType::IDENTITY),
    //(32, 48, false, TestActivationType::SINE, 16, false, TestActivationType::IDENTITY),
    //(32, 48, false, TestActivationType::SINE, 16, false, TestActivationType::SINE),
    //(32, 48, false, TestActivationType::CELU, 16, false, TestActivationType::CELU),

    //(32, 48, false, TestActivationType::IDENTITY, 32, false, TestActivationType::IDENTITY),
    //(32, 48, false, TestActivationType::SINE, 32, false, TestActivationType::IDENTITY),
    //(32, 48, false, TestActivationType::SINE, 32, false, TestActivationType::SINE),
    //(32, 48, false, TestActivationType::CELU, 32, false, TestActivationType::CELU)//,

    //TODO: Bias
    )
{
    bool skewSharedMemory = false;
    bool parallelWeightUpdate = false;
    SECTION("serial-noSkew")
    {
        parallelWeightUpdate = false;
        skewSharedMemory = false;
    }
    SECTION("parallel-noSkew")
    {
        parallelWeightUpdate = true;
        skewSharedMemory = false;
    }
    SECTION("parallel-skew")
    {
        parallelWeightUpdate = true;
        skewSharedMemory = true;
    }

    nlohmann::json cfg = {
        {"num_inputs", Channels0},
        {"num_outputs", Channels2},
        {"activation_specification", nlohmann::json::array({
            "qmlp/builtin-activations.json"
        }) },
        {"encodings", nlohmann::json::array({
            nlohmann::json::object({
                {"id", "Identity"},
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
        })},
        {"options", nlohmann::json::object({
            {"skew_shared_memory", skewSharedMemory},
            {"parallel_weight_update", parallelWeightUpdate}
        })}
    };
    std::filesystem::path current(__FILE__);
    auto root = current.parent_path().parent_path();
    auto configFolder = root / "network_configs";

    int N = 32;
    CUstream stream = nullptr;

    //create network
    QuickMLP::Instance().setCompileDebugMode(false);
    QuickMLP::Instance().setLogLevel(spdlog::level::info);

    auto network = std::make_shared<qmlp::FusedNetwork>(cfg, configFolder);
    INFO("network parameters: " << network->networkParameterCount());

    //create parameter tensor
    std::default_random_engine rng(42);  // NOLINT(cert-msc51-cpp)
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
    //run Eigen network
    EigenMatrixX outputEigenHost;
    {
        EigenMatrixX input = padInput(inputHost).transpose();
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
        outputEigenHost = removePadOutput(out1.transpose(), Channels2);
    }
    //compare
    COMPARE_TENSOR_AND_MATRIX(outputDevice, outputEigenHost);

    //DERIVATIVES

    //adjoint output
    EigenMatrixX adjOutputHost = EigenMatrixX::Random(N, network->channelsOut());
    adjOutputHost.setZero();
    //adjOutputHost(30, 15) = 1.0f; //this succeeds
    adjOutputHost(2, 14) = 2.0f; //this fails

    qmlp::Tensor adjOutputDevice(network->precisionOut(), { N, network->channelsOut() });
    toGpuTensor(adjOutputDevice, adjOutputHost);

    //first run Eigen
    EigenMatrixX adjInputEigen, adjWeights0Eigen, adjBias0Eigen, adjWeights1Eigen, adjBias1Eigen;
    {
        //forward
        EigenMatrixX input = padInput(inputHost).transpose();
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
        EigenMatrixX adjOut1 = padInput(adjOutputHost).transpose();
        //std::cout << "adjOut1 = " << adjOut1.block(0, 0, adjOut1.rows(), 1).transpose() << "\n";

        EigenMatrixX adjOutTemp1 = TestActivation<Activ2>::adjoint(outTemp1, adjOut1);
        adjBias1Eigen = adjOutTemp1.rowwise().sum();
        adjWeights1Eigen = (adjOutTemp1 * out0.transpose());
        EigenMatrixX adjOut0 = weights1.transpose() * adjOutTemp1;

        //std::cout << "adjOutTemp1^T =\n" << adjOutTemp1.transpose().eval().format(SmallFmt) << "\n";
        //std::cout << "out0 = \n" << out0.format(SmallFmt) << "\n";
        //std::cout << "adjOutTemp1 = " << adjOutTemp1.block(0, 0, adjOutTemp1.rows(), 1).transpose() << "\n";
        //std::cout << "adjOut0 = " << adjOut0.block(0, 0, adjOut0.rows(), 1).transpose() << "\n";

        EigenMatrixX adjOutTemp0 = TestActivation<Activ1>::adjoint(outTemp0, adjOut0);
        adjBias0Eigen = adjOutTemp0.rowwise().sum();
        adjWeights0Eigen = adjOutTemp0 * input.transpose();
        EigenMatrixX adjInput = weights0.transpose() * adjOutTemp0;

        //std::cout << "adjOutTemp0^T =\n" << adjOutTemp0.transpose().eval().format(SmallFmt) << "\n";
        //std::cout << "input = \n" << input.format(SmallFmt) << "\n";
        //std::cout << "adjOutTemp0 = " << adjOutTemp0.block(0, 0, adjOutTemp0.rows(), 1).transpose() << "\n";
        //std::cout << "adjInput = " << adjInput.block(0, 0, adjInput.rows(), 1).transpose() << "\n";

        adjInputEigen = removePadOutput(adjInput.transpose(), Channels0);
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
        int tmpSize = adjointWithFlags(
            qmlp::FusedNetwork::GRADIENTS_NETWORK_WEIGHTS);
        INFO("size of temporary memory: " << tmpSize);

        COMPARE_TENSOR_AND_MATRIX(
            network->networkParameter(0, false, Tensor::GRADIENTS),
            adjWeights0Eigen);
        COMPARE_TENSOR_AND_MATRIX(
            network->networkParameter(1, false, Tensor::GRADIENTS),
            adjWeights1Eigen);
        if constexpr (Bias1)
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

    //all derivatives
    {
        int tmpSize = adjointWithFlags(
            qmlp::FusedNetwork::GRADIENTS_NETWORK_WEIGHTS |
            qmlp::FusedNetwork::GRADIENTS_INPUT);
        INFO("size of temporary memory: " << tmpSize);

        COMPARE_TENSOR_AND_MATRIX(adjInputDevice, adjInputEigen);

        //std::cout << "adjWeights1-Eigen = \n" << adjWeights1Eigen.format(SmallFmt) << "\n";
        //EigenMatrixX adjWeights1Host = toEigenMatrix(network->networkParameter(1, false, Tensor::GRADIENTS));
        //std::cout << "adjWeights1-CUDA = \n" << adjWeights1Host.format(SmallFmt) << "\n";

        //std::cout << "adjWeights0-Eigen = \n" << adjWeights0Eigen.format(SmallFmt) << "\n";
        //EigenMatrixX adjWeights0Host = toEigenMatrix(network->networkParameter(0, false, Tensor::GRADIENTS));
        //std::cout << "adjWeights0-CUDA = \n" << adjWeights0Host.format(SmallFmt) << "\n";

        COMPARE_TENSOR_AND_MATRIX(
            network->networkParameter(0, false, Tensor::GRADIENTS),
            adjWeights0Eigen);
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

