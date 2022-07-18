#include "catch.hpp"

#include <qmlp/encoding_hashgrid.h>

#include "test_against_eigen.h"

using namespace qmlp;

TEST_CASE("Densegrid2D", "[encoding]")
{
    int channels = 4;
    int resolution = 8;
    int N = 16;

    auto grid = std::make_shared<EncodingHashGrid>(
        0, 2, 1, channels, -1, resolution, resolution, LayerCombinationMode::ADD, 
        std::vector<float>(), std::vector<float>());

    Tensor input = Tensor(Tensor::FLOAT, { N, grid->maxInputChannel() + 1 });
    Tensor parameter = Tensor(grid->parameterPrecision(Tensor::INFERENCE), { grid->parameterCount() });
    Tensor output = Tensor(Tensor::FLOAT, { N, grid->numOutputChannels() });

    std::default_random_engine rng(42);  // NOLINT(cert-msc51-cpp)
    input.rand_(-0.5f, +0.5f, rng);
    parameter.rand_(0, 1, rng);

    grid->forward(input, output, 0, parameter);
    CKL_SAFE_CALL(cudaDeviceSynchronize());

    EigenMatrixX inputEigen = tests::toEigenMatrix(input);
    EigenVectorX paramEigen = tests::toEigenVector(parameter);
    EigenMatrixX outputEigen = tests::toEigenMatrix(output);

    std::cout << "Input:\n" << inputEigen.format(tests::SmallFmt) << std::endl;
    std::cout << "Params:\n" << paramEigen.transpose().format(tests::SmallFmt) << std::endl;
    std::cout << "Output:\n" << outputEigen.format(tests::SmallFmt) << std::endl;
}