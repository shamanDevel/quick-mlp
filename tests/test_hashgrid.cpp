#include "catch.hpp"

#include <qmlp/encoding_hashgrid.h>
#include <qmlp/encoding_line_integration.h>

#include "test_against_eigen.h"

using namespace qmlp;

TEST_CASE("Densegrid2D", "[encoding]")
{
    int channels = 4;
    int resolution = 8;
    int N = 16;

    LayerCombinationMode combinationMode;
    SECTION("add")
    {
        combinationMode = LayerCombinationMode::ADD;
    }
    SECTION("concat")
    {
        combinationMode = LayerCombinationMode::CONCAT;
    }
    auto grid = std::make_shared<EncodingHashGrid>(
        0, 2, 2, channels, -1, resolution, resolution+2, combinationMode,
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

    Tensor adjOutput = Tensor(Tensor::FLOAT, { N, grid->numOutputChannels() });
    Tensor adjInput = Tensor(Tensor::FLOAT, { N, grid->maxInputChannel() + 1 });
    Tensor adjParam = Tensor(grid->parameterPrecision(Tensor::GRADIENTS), { grid->parameterCount() });

    adjOutput.rand_(-0.5f, +0.5f, rng);
    adjInput.zero_();
    adjParam.zero_();

    grid->adjoint(input, adjOutput, adjInput, 0, parameter, adjParam);
    CKL_SAFE_CALL(cudaDeviceSynchronize());

    EigenMatrixX adjInputEigen = tests::toEigenMatrix(adjInput);
    EigenVectorX adjParamEigen = tests::toEigenVector(adjParam);
    EigenMatrixX adjOutputEigen = tests::toEigenMatrix(adjOutput);

    std::cout << "Adjoint Output:\n" << adjOutputEigen.format(tests::SmallFmt) << std::endl;
    std::cout << "Adjoint Input:\n" << adjInputEigen.format(tests::SmallFmt) << std::endl;
    std::cout << "Adjoint Params:\n" << adjParamEigen.transpose().format(tests::SmallFmt) << std::endl;
}

TEST_CASE("LineIntegration3D", "[encoding]")
{
    int channels = 4;
    int resolution = 8;
    int N = 16;

    std::string combinationMode;
    SECTION("add")
    {
        combinationMode = "add";
    }
    SECTION("concat")
    {
        combinationMode = "concat";
    }

    using namespace nlohmann;
    auto config = R"json(
{
    "id": "LineIntegration",
    "start_in": 0,
    "n_in": 3,
    "volume": {
        "id": "HashGrid",
        "n_levels": 2,
        "n_features_per_level": 4,
        "log2_hashmap_size": 7,
        "min_resolution": 3,
        "max_resolution": 6,
        "combination_mode": "concat"
    },
    "stepsize": 0.01,
    "blending": "additive"
}
    )json"_json;
    config["volume"]["combination_mode"] = combinationMode;

    auto enc = EncodingFactory::Instance().create(config);
    REQUIRE(enc->id() == "LineIntegration");
    auto enc2 = std::dynamic_pointer_cast<EncodingLineIntegration>(enc);
    REQUIRE(enc2 != nullptr);

    REQUIRE(enc2->ndim() == 6);
    REQUIRE(enc2->maxInputChannel() == 5);
    REQUIRE(enc2->volume()->ndim() == 3);

    Tensor input = Tensor(Tensor::FLOAT, { N, enc2->maxInputChannel() + 1 });
    Tensor parameter = Tensor(enc2->parameterPrecision(Tensor::INFERENCE), { enc2->parameterCount() });
    Tensor output = Tensor(Tensor::FLOAT, { N, enc2->numOutputChannels() });

    std::default_random_engine rng(42);  // NOLINT(cert-msc51-cpp)
    input.rand_(-0.5f, +0.5f, rng);
    parameter.rand_(0, 1, rng);

    enc2->forward(input, output, 0, parameter);
    CKL_SAFE_CALL(cudaDeviceSynchronize());

    EigenMatrixX inputEigen = tests::toEigenMatrix(input);
    EigenVectorX paramEigen = tests::toEigenVector(parameter);
    EigenMatrixX outputEigen = tests::toEigenMatrix(output);

    std::cout << "Input:\n" << inputEigen.format(tests::SmallFmt) << std::endl;
    std::cout << "Params:\n" << paramEigen.transpose().format(tests::SmallFmt) << std::endl;
    std::cout << "Output:\n" << outputEigen.format(tests::SmallFmt) << std::endl;

    Tensor adjOutput = Tensor(Tensor::FLOAT, { N, enc2->numOutputChannels() });
    Tensor adjInput = Tensor(Tensor::FLOAT, { N, enc2->maxInputChannel() + 1 });
    Tensor adjParam = Tensor(enc2->parameterPrecision(Tensor::GRADIENTS), { enc2->parameterCount() });

    adjOutput.rand_(-0.5f, +0.5f, rng);
    adjInput.zero_();
    adjParam.zero_();

    enc2->adjoint(input, adjOutput, adjInput, 0, parameter, adjParam);
    CKL_SAFE_CALL(cudaDeviceSynchronize());

    EigenMatrixX adjInputEigen = tests::toEigenMatrix(adjInput);
    EigenVectorX adjParamEigen = tests::toEigenVector(adjParam);
    EigenMatrixX adjOutputEigen = tests::toEigenMatrix(adjOutput);

    std::cout << "Adjoint Output:\n" << adjOutputEigen.format(tests::SmallFmt) << std::endl;
    std::cout << "Adjoint Input:\n" << adjInputEigen.format(tests::SmallFmt) << std::endl;
    std::cout << "Adjoint Params:\n" << adjParamEigen.transpose().format(tests::SmallFmt) << std::endl;
}

