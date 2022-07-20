#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

#include "test_against_eigen.h"
#include "eigen_half.h"

QUICKMLP_NAMESPACE_BEGIN
namespace tests
{
    const Eigen::IOFormat SmallFmt = Eigen::IOFormat(3);
}
QUICKMLP_NAMESPACE_END

using namespace qmlp;
using namespace qmlp::tests;

TEMPLATE_TEST_CASE_SIG("test-against-eigen-1", "[eigen]",
    ((int Channels0, int Channels1, bool Bias1, TestActivationType Activ1),
        Channels0, Channels1, Bias1, Activ1),
    (16, 16, false, TestActivationType::SINE)//,
    //(16, 16, false, TestActivationType::RELU),
    //(16, 16, false, TestActivationType::IDENTITY),
    //(16, 32, false, TestActivationType::SINE),
    //(16, 32, false, TestActivationType::RELU),
    //(16, 32, false, TestActivationType::IDENTITY),
    //(48, 16, false, TestActivationType::SINE),
    //(48, 16, false, TestActivationType::RELU),
    //(48, 16, false, TestActivationType::IDENTITY),
    //(48, 32, false, TestActivationType::SINE),
    //(48, 32, false, TestActivationType::RELU),
    //(48, 32, false, TestActivationType::IDENTITY),
    //(16, 16, true, TestActivationType::SINE),
    //(16, 16, true, TestActivationType::RELU),
    //(16, 16, true, TestActivationType::IDENTITY),
    //(16, 32, true, TestActivationType::SINE),
    //(16, 32, true, TestActivationType::RELU),
    //(16, 32, true, TestActivationType::IDENTITY),
    //(48, 16, true, TestActivationType::SINE),
    //(48, 16, true, TestActivationType::RELU),
    //(48, 16, true, TestActivationType::IDENTITY),
    //(48, 32, true, TestActivationType::SINE),
    //(48, 32, true, TestActivationType::RELU),
    //(48, 32, true, TestActivationType::IDENTITY)
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
            })
        })}
    };
    std::filesystem::path current(__FILE__);
    auto root = current.parent_path().parent_path();
    auto configFolder = root / "network_configs";

    int N = 64+32;
    CUstream stream = nullptr;

    //create network
    auto network = std::make_shared<qmlp::FusedNetwork>(cfg, configFolder);
    WARN("network parameters: " << network->networkParameterCount());

    //create parameter tensor
    std::default_random_engine rng(42);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution<EigenScalar_t> distr;
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
    auto uniformGenerator = [&rng, &distr]() {return distr(rng); };
    EigenMatrixX inputHost = EigenMatrixX::NullaryExpr(N, network->channelsIn(), uniformGenerator);
    ////TEST
    //inputHost.setZero();
    //inputHost(1, 0) = 2.f;
    //copy to GPU
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
        EigenMatrixX outTemp0 = Bias1
            ? ((weights0 * input).colwise() + bias0).eval()
            : (weights0 * input).eval();
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
        EigenMatrixX outTemp0 = Bias1
            ? ((weights0 * input).colwise() + bias0).eval()
            : (weights0 * input).eval();
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);

        //backward
        EigenMatrixX adjOut0 = adjOutputHost.transpose();
        EigenMatrixX adjOutTemp0 = TestActivation<Activ1>::adjoint(outTemp0, adjOut0);
        adjBias0Eigen = adjOutTemp0.rowwise().sum();
        adjWeights0Eigen = (adjOutTemp0.cast<EigenHalf>() * input.cast<EigenHalf>().transpose()).cast<EigenScalar_t>();
        EigenMatrixX adjInput = weights0.transpose() * adjOutTemp0;

        std::cout << "adjOutTemp0 =\n" << adjOutTemp0.format(SmallFmt) << "\n";
        std::cout << "input^T = \n" << input.transpose().eval().format(SmallFmt) << "\n";
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
        std::cout.precision(3);
        INFO("size of temporary memory: " << tmpSize);
        COMPARE_TENSOR_AND_MATRIX(adjInputDevice, adjInputEigen);
    }

    //only weight derivatives
    {
        int tmpSize = adjointWithFlags(qmlp::FusedNetwork::GRADIENTS_NETWORK_WEIGHTS);
        INFO("size of temporary memory: " << tmpSize);
        COMPARE_TENSOR_AND_MATRIX(
            network->networkParameter(0, false, Tensor::GRADIENTS),
            adjWeights0Eigen);
        //if constexpr (Bias1)
        //{
        //    COMPARE_TENSOR_AND_VECTOR(
        //        network->networkParameter(0, true, Tensor::GRADIENTS),
        //        adjBias0Eigen);
        //}
    }
}

