#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

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
    template<>
    struct TestActivation<TestActivationType::IDENTITY>
    {
        static Eigen::MatrixXf forward(const Eigen::MatrixXf& x)
        {
            return x;
        }
        static Eigen::MatrixXf adjoint(const Eigen::MatrixXf& x, const Eigen::MatrixXf& adjz)
        {
            return adjz;
        }
    };

    Eigen::MatrixXf toEigenMatrix(const qmlp::Tensor& tensor)
    {
        REQUIRE(tensor.ndim() == 2);

        const void* dataDevice = tensor.rawPtr();
        std::vector<char> dataHost(tensor.numel() * tensor.bytesPerEntry());
        CKL_SAFE_CALL(cudaMemcpy(dataHost.data(), dataDevice, dataHost.size(), cudaMemcpyDeviceToHost));

        Eigen::MatrixXf matrix(tensor.size(0), tensor.size(1));
        if (tensor.precision() == qmlp::Tensor::FLOAT)
        {
            const float* data = reinterpret_cast<float*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i) for (int j = 0; j < tensor.size(1); ++j)
                matrix(i, j) = data[tensor.idx({ i,j })];
        }
        else if (tensor.precision() == qmlp::Tensor::HALF)
        {
            const half* data = reinterpret_cast<half*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i) for (int j = 0; j < tensor.size(1); ++j)
                matrix(i, j) = __half2float(data[tensor.idx({ i,j })]);
        }
        else
            throw std::runtime_error("Unknown precision");
        return matrix;
    }
    Eigen::VectorXf toEigenVector(const qmlp::Tensor& tensor)
    {
        REQUIRE(tensor.ndim() == 1);

        const void* dataDevice = tensor.rawPtr();
        std::vector<char> dataHost(tensor.numel() * tensor.bytesPerEntry());
        CKL_SAFE_CALL(cudaMemcpy(dataHost.data(), dataDevice, dataHost.size(), cudaMemcpyDeviceToHost));

        Eigen::VectorXf matrix(tensor.size(0));
        if (tensor.precision() == qmlp::Tensor::FLOAT)
        {
            const float* data = reinterpret_cast<float*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i)
                matrix[i] = data[tensor.idx({ i })];
        }
        else if (tensor.precision() == qmlp::Tensor::HALF)
        {
            const half* data = reinterpret_cast<half*>(dataHost.data());
            for (int i = 0; i < tensor.size(0); ++i)
                matrix[i] = __half2float(data[tensor.idx({ i })]);
        }
        else
            throw std::runtime_error("Unknown precision");
        return matrix;
    }
    void toGpuTensor(qmlp::Tensor& dst, const Eigen::MatrixXf& src)
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
    void compareEigen(const Eigen::MatrixXf& actual, const Eigen::MatrixXf& expected)
    {
        REQUIRE(actual.rows() == expected.rows());
        REQUIRE(actual.cols() == expected.cols());
        for (int i = 0; i < actual.rows(); ++i) for (int j = 0; j < actual.cols(); ++j)
        {
            INFO("i=" << i << ", j=" << j);
            float va = actual(i, j);
            float ve = expected(i, j);
            INFO("actual=" << va << ", expected=" << ve);
            REQUIRE(sgn(va) == sgn(ve));
            float relDiff = va / ve;
            const float eps = 5;
            REQUIRE((1/eps < relDiff && relDiff < eps));
        }
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
    Eigen::MatrixXf inputHost = Eigen::MatrixXf::Random(N, network->channelsIn());
    toGpuTensor(inputDevice, inputHost);

    //INFERENCE
    //run cuda network
    network->inference(inputDevice, outputDevice, stream);
    CKL_SAFE_CALL(cudaDeviceSynchronize());
    Eigen::MatrixXf outputCudaHost = toEigenMatrix(outputDevice);
    //run Eigen network
    Eigen::MatrixXf outputEigenHost;
    {
        auto input = inputHost.transpose();
        Eigen::MatrixXf weights0 = toEigenMatrix(network->networkParameter(0, false, qmlp::Tensor::INFERENCE));
        Eigen::VectorXf bias0 = toEigenVector(network->networkParameter(0, true, qmlp::Tensor::INFERENCE));
        Eigen::MatrixXf weights1 = toEigenMatrix(network->networkParameter(1, false, qmlp::Tensor::INFERENCE));
        Eigen::VectorXf bias1 = toEigenVector(network->networkParameter(1, true, qmlp::Tensor::INFERENCE));
        Eigen::MatrixXf outTemp0 = (weights0 * input).colwise() + bias0;
        Eigen::MatrixXf out0 = TestActivation<Activ1>::forward(outTemp0);
        Eigen::MatrixXf outTemp1 = (weights1 * out0).colwise() + bias1;
        Eigen::MatrixXf out1 = TestActivation<Activ2>::forward(outTemp1);
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
    Eigen::MatrixXf adjOutputHost = Eigen::MatrixXf::Random(N, network->channelsOut());
    qmlp::Tensor adjOutputDevice(network->precisionOut(), { N, network->channelsOut() });
    toGpuTensor(adjOutputDevice, adjOutputHost);

    //first run Eigen
    Eigen::MatrixXf adjInputEigen, adjWeights0Eigen, adjBias0Eigen, adjWeights1Eigen, adjBias1Eigen;
    {
        //forward
        auto input = inputHost.transpose();
        Eigen::MatrixXf weights0 = toEigenMatrix(network->networkParameter(0, false, qmlp::Tensor::INFERENCE));
        Eigen::VectorXf bias0 = toEigenVector(network->networkParameter(0, true, qmlp::Tensor::INFERENCE));
        Eigen::MatrixXf weights1 = toEigenMatrix(network->networkParameter(1, false, qmlp::Tensor::INFERENCE));
        Eigen::VectorXf bias1 = toEigenVector(network->networkParameter(1, true, qmlp::Tensor::INFERENCE));
        Eigen::MatrixXf outTemp0 = (weights0 * input).colwise() + bias0;
        Eigen::MatrixXf out0 = TestActivation<Activ1>::forward(outTemp0);
        Eigen::MatrixXf outTemp1 = (weights1 * out0).colwise() + bias1;
        Eigen::MatrixXf out1 = TestActivation<Activ2>::forward(outTemp1);
        //backward
        Eigen::MatrixXf adjOut1 = adjOutputHost.transpose();
        std::cout << "adjOut1 = " << adjOut1.block(0, 0, adjOut1.rows(), 1).transpose() << "\n";

        Eigen::MatrixXf adjOutTemp1 = TestActivation<Activ2>::adjoint(outTemp1, adjOut1);
        std::cout << "adjOutTemp1 = " << adjOutTemp1.block(0, 0, adjOutTemp1.rows(), 1).transpose() << "\n";
        adjBias1Eigen = adjOutTemp1.rowwise().sum();
        adjWeights1Eigen = adjOutTemp1 * out0.transpose();
        Eigen::MatrixXf adjOut0 = weights1.transpose() * adjOutTemp1;
        std::cout << "adjOut0 = " << adjOut0.block(0, 0, adjOut0.rows(), 1).transpose() << "\n";

        Eigen::MatrixXf adjOutTemp0 = TestActivation<Activ2>::adjoint(outTemp0, adjOut0);
        std::cout << "adjOutTemp0 = " << adjOutTemp0.block(0, 0, adjOutTemp0.rows(), 1).transpose() << "\n";
        adjBias0Eigen = adjOutTemp0.rowwise().sum();
        adjWeights0Eigen = adjOutTemp0 * input.transpose();
        Eigen::MatrixXf adjInput = weights0.transpose() * adjOutTemp0;
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
    const auto compareTensorAndMatrix = [](const qmlp::Tensor& actual, const Eigen::MatrixXf& expected, const char* line)
    {
        INFO("test: " << line);
        Eigen::MatrixXf actualHost = toEigenMatrix(actual);
        INFO("actual (CUDA):\n" << actualHost);
        INFO("expected (Eigen):\n" << expected);
        compareEigen(actualHost, expected);
    };
#define COMPARE_TENSOR_AND_MATRIX(...) compareTensorAndMatrix(__VA_ARGS__, CKL_STR(__VA_ARGS__))
    const auto compareTensorAndVector = [](const qmlp::Tensor& actual, const Eigen::VectorXf& expected, const char* line)
    {
        INFO("test: " << line);
        Eigen::VectorXf actualHost = toEigenVector(actual);
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

