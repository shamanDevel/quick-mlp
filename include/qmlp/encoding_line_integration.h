#pragma once

#include "common.h"
#include "iencoding.h"
#include <nlohmann/json.hpp>

#include "kernels/encoding_line_integration_config.cuh"

QUICKMLP_NAMESPACE_BEGIN

/**
 * Line Integration.
 * Network encoding that performs line integration along a trained volume
 * and passes the blended feature vector to the network.
 *
 * The backing volume must be an instance of IVolumetricEncoding.
 * Currently implemented:
 *  - HashGrid
 *
 * The step size is in world space!
 *
 * The input is interpreted as a concatenation of ray start + ray direction.
 * Therefore, if the dimension is 3, 6 input channels are read.
 *
 * The blending is one of:
 *  - "additive": multiple the interpolated values with the step size and sum them up
 *  - "average": compute the average of all samples along the ray
 */
class EncodingLineIntegration : public IVolumetricEncoding
{
    int startChannel_;
    int dimension_;
    std::shared_ptr<IVolumetricEncoding> volume_;
    float stepsizeWorld_;
    kernel::LineIntegrationBlendingMode blendingMode_;

public:
    EncodingLineIntegration(int start_channel, int dimension,
        std::shared_ptr<IVolumetricEncoding> volume, float stepsize, 
        kernel::LineIntegrationBlendingMode blendingMode);
    EncodingLineIntegration(const nlohmann::json& cfg);

    [[nodiscard]] nlohmann::json toJson() const override;
    static std::string ID();
    [[nodiscard]] std::string id() const override;

    [[nodiscard]] int maxInputChannel() const override;
    [[nodiscard]] int numOutputChannels() const override;
    [[nodiscard]] std::string qualifiedName() const override;
    void fillCode(std::stringstream& code) const override;

    [[nodiscard]] int ndim() const override;
    [[nodiscard]] BoundingBoxVector_t boundingBoxMin() const override;
    [[nodiscard]] BoundingBoxVector_t boundingBoxSize() const override;
    [[nodiscard]] BoundingBoxVector_t boundingBoxInvSize() const override;

    [[nodiscard]] bool hasParameters() const override;
    [[nodiscard]] std::string parameterName() const override;
    [[nodiscard]] Tensor::Precision parameterPrecision(Tensor::Usage usage) const override;
    [[nodiscard]] int parameterCount() const override;
    void setParameter(const Tensor& tensor, Tensor::Usage usage) override;
    [[nodiscard]] int fillParameterMemory(char* dst, int dstSize) override;
    void fillParameterConstant(const std::string& constantName, const ckl::KernelFunction& function,
        CUstream stream) override;
    void zeroGradients() override;
};

QUICKMLP_NAMESPACE_END
