#pragma once

#include <cuda.h>

#include "common.h"
#include "tensor.h"
#include "activation.h"
#include "iencoding.h"
#include <ckl/kernel_loader.h>

QUICKMLP_NAMESPACE_BEGIN
class FusedNetwork : NonAssignable
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
    int networkInputPadding_;
    std::vector<IEncoding_ptr> encodings_;
    std::vector<LayerSpecification> layers_;

    int numParameters_;
    Tensor parametersInference_;
    Tensor parametersGradients_;

    //CUstreams for training

public:
    //Size of the matrix fragments (hardware limitation of the tensor cores)
    //All inner sizes must be multiples of this.
    static constexpr int MATRIX_SIZE = 16;

    //Maximal shared memory on the hardware I'm using (48kB)
    static constexpr int MAX_SHARED_MEMORY_BYTES = 48 * 1024;

    /**
     * Constructs the inner network from the Json configuration 'cfg',
     * an array describing the individual layers.
     */
    FusedNetwork(const nlohmann::json& cfg, const std::filesystem::path& parent);

    [[nodiscard]] const std::vector<IEncoding_ptr>& encodings() const { return encodings_; }
    [[nodiscard]] int numEncodings() const { return encodings_.size(); }
    [[nodiscard]] IEncoding_ptr encoding(int idx);

    [[nodiscard]] int channelsIn() const { return channelsIn_; }
    [[nodiscard]] int channelsOut() const { return channelsOut_; }
    [[nodiscard]] Tensor::Precision precisionIn() const { return Tensor::FLOAT; }
    [[nodiscard]] Tensor::Precision precisionOut() const { return Tensor::FLOAT; }
    [[nodiscard]] int numLayers() const { return layers_.size(); }

    [[nodiscard]] int networkParameterCount() const;

    [[nodiscard]] Tensor::Precision networkParameterPrecision(Tensor::Usage usage) const;

    /**
     * Sets the underlying parameter to the specified tensor.
     * \param tensor the 1D tensor with \ref parameterCount() parameter of
     *   type parameterPrecision(Usage)
     * \param usage the usage for the parameters
     */
    void setNetworkParameter(const Tensor& tensor, Tensor::Usage usage);

    /**
     * \brief Returns the slice of the parameter corresponding
     * to the weight matrix (bias=false) or bias vector (bias=true)
     * of the specified layer 'layer'.
     *
     * Important: The resulting tensor <b>shared</b> the memory of the
     * tensor set by \ref setNetworkParameter().
     * This method is used during testing to validate the network
     * against PyTorch.
     *
     * \param layer the layer index in [0, numLayers()-1]
     * \param bias true->return bias vector; false->return weight matrix
     * \param usage inference or gradient
     */
    Tensor networkParameter(int layer, bool bias, Tensor::Usage usage);
    
    /**
     * \brief Performs inference of the network
     * \param input the input data of shape (N, channelsIn) of type precisionIn()
     * \param output the output data of shape (N, channelsOut) of type precisionIn()
     * \param stream the cuda stream where the kernel is emplaced onto
     */
    void inference(const Tensor& input, Tensor& output, CUstream stream);

    //TODO: training code

private:

    /**
     * Writes the parameters into the constant field denoted by 'constantName'.
     * This field is then later passed to the evaluation kernel.
     * \see ckl::KernelFunction::fillConstantMemoryAsync
     */
    void fillNetworkConstant(
        const std::string& constantName, const ckl::KernelFunction& function, CUstream stream);


};

QUICKMLP_NAMESPACE_END
