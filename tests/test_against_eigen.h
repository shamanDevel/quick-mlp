#pragma once

#include "catch.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include <qmlp/fused_network.h>

typedef Eigen::half EigenScalar_t;
typedef Eigen::MatrixX<EigenScalar_t> EigenMatrixX;
typedef Eigen::VectorX<EigenScalar_t> EigenVectorX;

QUICKMLP_NAMESPACE_BEGIN
namespace tests{
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
            return x.cwiseMax(EigenScalar_t(0.0f));
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

    inline EigenMatrixX toEigenMatrix(const qmlp::Tensor& tensor)
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
    inline EigenVectorX toEigenVector(const qmlp::Tensor& tensor)
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
    inline void toGpuTensor(qmlp::Tensor& dst, const EigenMatrixX& src)
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
    inline void compareEigen(const EigenMatrixX& actual, const EigenMatrixX& expected)
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

    inline int adjointWithFlags(FusedNetwork_ptr network, const Tensor& inputDevice, Tensor& outputDevice,
        Tensor& adjInputDevice, Tensor& adjOutputDevice, int flags, CUstream stream=nullptr)
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
    }
    inline void compareTensorAndMatrix(const qmlp::Tensor& actual, const EigenMatrixX& expected, const char* line)
    {
        INFO("test: " << line);
        EigenMatrixX actualHost = toEigenMatrix(actual);
        INFO("actual (CUDA):\n" << actualHost);
        INFO("expected (Eigen):\n" << expected);
        compareEigen(actualHost, expected);
    }
#define COMPARE_TENSOR_AND_MATRIX(...) compareTensorAndMatrix(__VA_ARGS__, CKL_STR(__VA_ARGS__))
    inline void compareTensorAndVector(const qmlp::Tensor& actual, const EigenVectorX& expected, const char* line)
    {
        INFO("test: " << line);
        EigenVectorX actualHost = toEigenVector(actual);
        INFO("actual (CUDA):\n" << actualHost);
        INFO("expected (Eigen):\n" << expected);
        compareEigen(actualHost, expected);
    }
#define COMPARE_TENSOR_AND_VECTOR(...) compareTensorAndVector(__VA_ARGS__, CKL_STR(__VA_ARGS__))
}
QUICKMLP_NAMESPACE_END
