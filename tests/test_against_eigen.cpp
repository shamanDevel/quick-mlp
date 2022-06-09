#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

typedef Eigen::half EigenScalar_t;
typedef Eigen::MatrixX<EigenScalar_t> EigenMatrixX;
typedef Eigen::VectorX<EigenScalar_t> EigenVectorX;

namespace {
    enum class TestActivationType
    {
        RELU, SINE, IDENTITY
    };
    //name for the config file
    static const char* TestActivationConfigName[] = {
        "relu", "sine", "identity"
    };
    template<TestActivationType T> struct TestActivation;
    template<>
    struct TestActivation<TestActivationType::RELU>
    {
        static EigenMatrixX forward(const EigenMatrixX& x)
        {
            return x.cwiseMin(EigenScalar_t(0.0f));
        }
        static EigenMatrixX adjoint(const EigenMatrixX& x, const EigenMatrixX& adjz)
        {
            return ((x.array() > EigenScalar_t(0.f)).cast<EigenScalar_t>() * adjz.array()).matrix();
        }
    };
    template<>
    struct TestActivation<TestActivationType::SINE>
    {
        static EigenMatrixX forward(const EigenMatrixX& x)
        {
            return x.array().sin().matrix();
        }
        static EigenMatrixX adjoint(const EigenMatrixX& x, const EigenMatrixX& adjz)
        {
            return (x.array().cos() * adjz.array()).matrix();
        }
    };
    template<>
    struct TestActivation<TestActivationType::IDENTITY>
    {
        static EigenMatrixX forward(const EigenMatrixX& x)
        {
            return x;
        }
        static EigenMatrixX adjoint(const EigenMatrixX& x, const EigenMatrixX& adjz)
        {
            return adjz;
        }
    };

    EigenMatrixX toEigenMatrix(const qmlp::Tensor& tensor)
    {
        REQUIRE(tensor.ndim() == 2);

        const void* dataDevice = tensor.rawPtr();
        std::vector<char> dataHost(tensor.numel() * tensor.bytesPerEntry());
        CKL_SAFE_CALL(cudaMemcpy(dataHost.data(), dataDevice, dataHost.size(), cudaMemcpyDeviceToHost));

        EigenMatrixX matrix(tensor.size(0), tensor.size(1));
        if (tensor.precision() == qmlp::Tensor::FLOAT)
        {
            const float* data = reinterpret_cast<float*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i) for (int j = 0; j < tensor.size(1); ++j)
                matrix(i, j) = EigenScalar_t(data[tensor.idx({ i,j })]);
        }
        else if (tensor.precision() == qmlp::Tensor::HALF)
        {
            const half* data = reinterpret_cast<half*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i) for (int j = 0; j < tensor.size(1); ++j)
                matrix(i, j) = EigenScalar_t(__half2float(data[tensor.idx({ i,j })]));
        }
        else
            throw std::runtime_error("Unknown precision");
        return matrix;
    }
    EigenVectorX toEigenVector(const qmlp::Tensor& tensor)
    {
        REQUIRE(tensor.ndim() == 1);

        const void* dataDevice = tensor.rawPtr();
        std::vector<char> dataHost(tensor.numel() * tensor.bytesPerEntry());
        CKL_SAFE_CALL(cudaMemcpy(dataHost.data(), dataDevice, dataHost.size(), cudaMemcpyDeviceToHost));

        EigenVectorX matrix(tensor.size(0));
        if (tensor.precision() == qmlp::Tensor::FLOAT)
        {
            const float* data = reinterpret_cast<float*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i)
                matrix[i] = EigenScalar_t(data[tensor.idx({ i })]);
        }
        else if (tensor.precision() == qmlp::Tensor::HALF)
        {
            const half* data = reinterpret_cast<half*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i)
                matrix[i] = EigenScalar_t(__half2float(data[tensor.idx({ i })]));
        }
        else
            throw std::runtime_error("Unknown precision");
        return matrix;
    }
    void toGpuTensor(qmlp::Tensor& dst, const EigenMatrixX& src)
    {
        REQUIRE(dst.ndim() == 2);
        REQUIRE(dst.size(0) == src.rows());
        REQUIRE(dst.size(1) == src.cols());

        std::vector<char> dataHost(dst.numel() * dst.bytesPerEntry());
        if (dst.precision() == qmlp::Tensor::FLOAT)
        {
            float* data = reinterpret_cast<float*>(dataHost.data());
            for (int i = 0; i < dst.size(0); ++i) for (int j = 0; j < dst.size(1); ++j)
                data[dst.idx({ i,j })] = src(i, j);
        }
        else if (dst.precision() == qmlp::Tensor::HALF)
        {
            half* data = reinterpret_cast<half*>(dataHost.data());
            for (int i = 0; i < dst.size(0); ++i) for (int j = 0; j < dst.size(1); ++j)
                data[dst.idx({ i,j })] = __float2half(src(i, j));
        }
        else
            throw std::runtime_error("Unknown precision");

        CKL_SAFE_CALL(cudaMemcpy(dst.rawPtr(), dataHost.data(), dataHost.size(), cudaMemcpyHostToDevice));
    }

    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
    void compareEigen(const EigenMatrixX& actual, const EigenMatrixX& expected)
    {
        REQUIRE(actual.rows() == expected.rows());
        REQUIRE(actual.cols() == expected.cols());
        static const float REL_ERROR = 0.1f; //10%
        static const float ABS_ERROR = 1e-3f;
        static const float ALLOWED_EXCEED = 0.01f; //1%
        int numExceed = 0;
        for (int i = 0; i < actual.rows(); ++i) for (int j = 0; j < actual.cols(); ++j)
        {
            auto va = static_cast<float>(actual(i, j));
            auto ve = static_cast<float>(expected(i, j));
            bool correct = va == Approx(ve).epsilon(REL_ERROR).margin(ABS_ERROR);
            if (!correct)
            {
                numExceed++;
                UNSCOPED_INFO("i=" << i << ", j=" << j << " => actual=" << va << ", expected=" << ve);
            }
        }
        float exceedFraction = numExceed / static_cast<float>(expected.size());
        REQUIRE(exceedFraction < ALLOWED_EXCEED);
    }
}

TEMPLATE_TEST_CASE_SIG("test-agaist-eigen", "[eigen]", 
    ((int Channels0, int Channels1, TestActivationType Activ1, int Channels2, TestActivationType Activ2),
        Channels0, Channels1, Activ1, Channels2, Activ2),
    (16, 16, TestActivationType::SINE, 16, TestActivationType::IDENTITY)
    //(16, 16, TestActivationType::SINE, 16, TestActivationType::SINE),
    //(16, 16, TestActivationType::RELU, 16, TestActivationType::IDENTITY),
    //(16, 16, TestActivationType::SINE, 32, TestActivationType::SINE),
    //(16, 16, TestActivationType::RELU, 32, TestActivationType::IDENTITY),
    //(32, 48, TestActivationType::SINE, 16, TestActivationType::SINE),
    //(32, 48, TestActivationType::RELU, 16, TestActivationType::IDENTITY),
    //(32, 48, TestActivationType::SINE, 32, TestActivationType::SINE),
    //(32, 48, TestActivationType::RELU, 32, TestActivationType::IDENTITY)
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
        EigenMatrixX outTemp0 = (weights0 * input).colwise() + bias0;
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);
        EigenMatrixX outTemp1 = (weights1 * out0).colwise() + bias1;
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
        EigenMatrixX outTemp0 = (weights0 * input).colwise() + bias0;
        EigenMatrixX out0 = TestActivation<Activ1>::forward(outTemp0);
        EigenMatrixX outTemp1 = (weights1 * out0).colwise() + bias1;
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

        EigenMatrixX adjOutTemp0 = TestActivation<Activ2>::adjoint(outTemp0, adjOut0);
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
        adjInputDevice.zero_();
        size_t s = network->forwardMemory(inputDevice, flags);
        INFO("allocating " << s << " bytes as temporary memory");
        void* tmpMem;
        CKL_SAFE_CALL(cudaMalloc(&tmpMem, s));
        network->forward(inputDevice, outputDevice, tmpMem, stream);
        network->adjoint(inputDevice, adjOutputDevice, flags, adjInputDevice, tmpMem, stream);
        CKL_SAFE_CALL(cudaFree(tmpMem));
        return s;
    };
    const auto compareTensorAndMatrix = [](const qmlp::Tensor& actual, const EigenMatrixX& expected, const char* line)
    {
        INFO("test: " << line);
        EigenMatrixX actualHost = toEigenMatrix(actual);
        INFO("actual (CUDA):\n" << actualHost);
        INFO("expected (Eigen):\n" << expected);
        compareEigen(actualHost, expected);
    };
#define COMPARE_TENSOR_AND_MATRIX(...) compareTensorAndMatrix(__VA_ARGS__, CKL_STR(__VA_ARGS__))
    const auto compareTensorAndVector = [](const qmlp::Tensor& actual, const EigenVectorX& expected, const char* line)
    {
        INFO("test: " << line);
        EigenVectorX actualHost = toEigenVector(actual);
        INFO("actual (CUDA):\n" << actualHost);
        INFO("expected (Eigen):\n" << expected);
        compareEigen(actualHost, expected);
    };
#define COMPARE_TENSOR_AND_VECTOR(...) compareTensorAndVector(__VA_ARGS__, CKL_STR(__VA_ARGS__))

    //only input derivatives
    {
        int tmpSize = adjointWithFlags(qmlp::FusedNetwork::GRADIENTS_INPUT);
        INFO("size of temporary memory: " << tmpSize);
        COMPARE_TENSOR_AND_MATRIX(adjInputDevice, adjInputEigen);
    }
}

