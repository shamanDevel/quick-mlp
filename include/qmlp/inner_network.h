#pragma once

#include <cuda.h>

#include "common.h"
#include "tensor.h"
#include "activation.h"
#include <ckl/kernel_loader.h>

QUICKMLP_NAMESPACE_BEGIN
class InnerNetwork
{
    struct LayerSpecification
    {
        //specs from Json
        int channelsOut;
        bool useBias;
        //Either a single entry, used for every channel
        //Or an array of size channelsOut, separate activations per channel.
        //This is useful in the last layer
        std::vector<Activation_ptr> activations;

        //cached parameter for easy access
        int channelsIn;
        //prefix sum, offset into parameter array
        int weightsStart;
        int biasStart;
    };

    int channelsIn_;
    int channelsOut_;
    std::vector<LayerSpecification> layers_;

    int numParameters_;
    Tensor parametersInference_;
    Tensor parametersGradients_;

    //

public:

    /**
     * Constructs the inner network from the Json configuration 'cfg',
     * an array describing the individual layers.
     */
    InnerNetwork(const nlohmann::json& cfg, int inputChannels, ActivationFactory_ptr activations);

    /**
     * Returns the fully-qualified name of the parameter structure.
     *
     */
    [[nodiscard]] virtual std::string parameterName() const { return ""; }

    [[nodiscard]] int parameterCount() const;

    [[nodiscard]] Tensor::Precision parameterPrecision(Tensor::Usage usage) const;

    /**
     * Sets the underlying parameter to the specified tensor.
     * \param tensor the 1D tensor with \ref parameterCount() parameter of
     *   type parameterPrecision(Usage)
     * \param usage the usage for the parameters
     */
    void setParameter(const Tensor& tensor, Tensor::Usage usage);

    /**
     * Writes the parameters into the constant field denoted by 'constantName'.
     * This field is then later passed to the evaluation kernel.
     * \see ckl::KernelFunction::fillConstantMemoryAsync
     */
    void fillParameterConstant(
        const std::string& constantName, const ckl::KernelFunction& function, CUstream stream);


};

QUICKMLP_NAMESPACE_END
