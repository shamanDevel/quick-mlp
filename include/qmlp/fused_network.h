#pragma once

#include <cuda.h>
#include <random>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "tensor.h"
#include "activation.h"
#include "iencoding.h"
#include <ckl/kernel_loader.h>

QUICKMLP_NAMESPACE_BEGIN
class FusedNetwork : NonAssignable
{
public:
    /**
     * additive flags specifying what gradients to compute in the adjoint pass
     */
    enum AdjointMode
    {
        //Compute gradients with respect to the network input
        GRADIENTS_INPUT = 0x1,
        //Compute gradients with respect to the network weights
        GRADIENTS_NETWORK_WEIGHTS = 0x2,
        //Compute gradients with respect to the input encodings
        GRADIENTS_INPUT_ENCODINGS = 0x4,

        ADJOINT_MODE_MAX_COMBINATIONS = 8
    };
    typedef int AdjointModeFlags;

private:
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

    //compiled kernels
    struct CompiledKernel
    {
        ckl::KernelFunction fun;
        int blockSize;
        int sharedMemorySize;
        std::vector<std::string> encodingIdx2ConstantName;
        std::unordered_map<IEncoding_ptr, std::string> encodingPtr2ConstantName;
    };
    std::optional<CompiledKernel> inferenceKernel_;
    std::optional<CompiledKernel> forwardKernel_;

    struct AdjointLayerInfo
    {
        //multiply with numel to get the offset into the adjoint-post-matmul array
        int offsetAdjOut;
        //multiply with numel to get the offset into the pre-activation array.
        // for weight derivatives, the activation has to be applied on-the-fly again
        //In the first layer, this value is -1, as the input encodings are needed again.
        int offsetIn;

        CompiledKernel adjWeightKernel;
        CompiledKernel adjBiasKernel;
    };
    struct AdjointKernel
    {
        CompiledKernel adjoint;
        std::vector<AdjointLayerInfo> layers;
        int perEntryAdjointMemoryBytes;
    };
    std::optional<AdjointKernel> backwardKernel_[ADJOINT_MODE_MAX_COMBINATIONS];
    //The number of bytes for storing the intermediate forward values.
    //If network weights are optimized, this amount needs to be doubled
    //because adjoint variables of those have to be stored in global memory
    //for computing the network weights afterwards using CUTLASS
    int perEntryForwardMemoryBytes_;

    //CUstreams for training
    std::vector<CUstream> trainingStreams_;
    std::vector<CUevent> trainingEvents_;

public:
    //Size of the matrix fragments (hardware limitation of the tensor cores)
    //All inner sizes must be multiples of this.
    static constexpr int MATRIX_SIZE = 16;

    //Maximal shared memory on the hardware I'm using (48kB)
    static constexpr int MAX_SHARED_MEMORY_BYTES = 48 * 1024;

    /**
     * \brief Constructs the inner network from the Json configuration 'cfg',
     * an array describing the individual layers.
     *
     * After constructing, the configuration is fixed and immutable.
     * This way, the kernels can be compiled once and then cached.
     */
    FusedNetwork(const nlohmann::json& cfg, const std::filesystem::path& parent);
    ~FusedNetwork();

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
     * Initializes the inference weights of the networks to random values.
     * This follows the default initialization routine from PyTorch.
     * The results are written into the inference tensor specified by
     * \ref setNetworkParameter().
     */
    void initializeInferenceParameters(std::default_random_engine& rng);

private:
    Tensor networkParameter(int layer, bool bias, void* rawPtr, Tensor::Precision precision);

public:
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

    /**
     * \brief Computes the memory in bytes required as temporary memory
     * to store the intermediate results of the forward method.
     * \param numElements the number of elements to process
     * \param adjointFlags additive flags of AdjointMode specifying what to differentiate
     * \return the number of bytes required to store the intermediate results
     *   to process the specified inputs.
     */
    size_t forwardMemory(int numElements, AdjointModeFlags adjointFlags);

    /**
     * \brief Computes the memory in bytes required as temporary memory
     * to store the intermediate results of the forward method.
     * \param input the input to the forward method
     * \param adjointFlags additive flags of AdjointMode specifying what to differentiate
     * \return the number of bytes required to store the intermediate results
     *   to process the specified inputs.
     */
    size_t forwardMemory(const Tensor& input, AdjointModeFlags adjointFlags);
    
    /**
     * \brief Inference / forward pass with storing intermediate results
     *  to allow for gradient propagation.
     * After this method, you can compute the gradients via \ref adjoint()
     *
     * \param input the inputs to the network of shape (B, Cin)
     * \param output the output of the network of shape (B, Cout)
     * \param tmpMemory temporary memory on the GPU with \ref forwardMemory(Tensor) bytes
     * \param stream the CUDA stream where the kernel is enqueued.
     */
    void forward(const Tensor& input, Tensor& output, void* tmpMemory, CUstream stream);

    /**
     * Zeros the gradients for the weights of the network and the
     * gradients of the input encodings.
     */
    void zeroGradients();

    /**
    * \brief Computes the memory in bytes required as temporary memory
     * during the adjoint pass. This memory is only used while computing
     * the adjoint gradients and does not need to be stored persistently.
    * \param numElements the number of elements to process
    * \param adjointFlags additive flags of AdjointMode specifying what to differentiate
    * \return the number of bytes required to store the intermediate results
    *   to process the specified inputs.
    */
    size_t adjointMemory(int numElements, AdjointModeFlags adjointFlags);

    /**
     * \brief Computes the memory in bytes required as temporary memory
     * during the adjoint pass. This memory is only used while computing
     * the adjoint gradients and does not need to be stored persistently.
     * \param input the network inputs for determing the size
     * \param adjointFlags additive flags of AdjointMode specifying what to differentiate
     * \return the number of bytes required to store the intermediate results
     *   to process the specified inputs.
     */
    size_t adjointMemory(const Tensor& input, AdjointModeFlags adjointFlags);

    /**
     * \brief Performs the backpropagation / adjoint differentiation.
     * This method follows a call to \ref forward() and computes the gradients.
     * The gradients for the network weights are added to the gradient tensor specified
     * by \ef setNetworkParameter() with <code>usage=Usage::Gradient</code>,
     * same for the input encodings.
     *
     * Additionally, derivatives with respect to the inputs can be computed when
     * \c adjInput is defined.
     *
     * \param input the input tensor from the forward pass
     * \param adjOutput the adjoint of the output
     * \param adjointFlags additive flags of AdjointMode specifying what to differentiate
     * \param adjInput [optional] accumulates gradients with respect to the inputs
     * \param tmpMemoryForward temporary memory from the forward pass,
     *   see \ref forwardMemory(Tensor, AdjointModeFlags)
     * \param tmpMemoryAdjoint temporary memory for intermediate results
     *   within the adjoint propagation. See \ref adjointMemory(Tensor, AdjointModeFlags).
     *   Can be \c nullptr if \c adjointMemory() also requests zero bytes.
     * \param stream the CUDA stream where the kernel is enqueued.
     */
    void adjoint(const Tensor& input, const Tensor& adjOutput, AdjointModeFlags adjointFlags,
        Tensor& adjInput, const void* tmpMemoryForward, void* tmpMemoryAdjoint, CUstream stream);
    
    /**
     * For debugging, clears all cached kernels.
     */
    void clearCachedKernels();

private:

    void compileInferenceKernel();
    void compileForwardKernel();
    void compileBackwardKernel(int flags);

    std::string constantNameForEncoding(IEncoding_ptr e, int encodingIdx);
    void fillEncodingsAndActivations(
        std::string& codeTemplate, std::vector<std::string>& encodingConstantNames);

    ///**
    // * Writes the parameters into the constant field denoted by 'constantName'.
    // * This field is then later passed to the evaluation kernel.
    // * \see ckl::KernelFunction::fillConstantMemoryAsync
    // */
    //void fillNetworkConstant(
    //    const std::string& constantName, const ckl::KernelFunction& function, CUstream stream);


};
typedef std::shared_ptr<FusedNetwork> FusedNetwork_ptr;

QUICKMLP_NAMESPACE_END
