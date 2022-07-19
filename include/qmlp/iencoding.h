#pragma once

#include "common.h"

#include <vector>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <optional>
#include <ckl/kernel_loader.h>
#include <nlohmann/json.hpp>

#include "tensor.h"

QUICKMLP_NAMESPACE_BEGIN

/**
 * Defines the input encoding
 *
 * Synapsis, what the implementations must provide:
 * <code>
 * struct NameOfEncoding
 * {
 *     template<typename I, typename O>
 *     static __device__ void forward(const I input, O* output [, const param-type& param-name ]);
 *
 *     template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O, typename AdjI>
 *     static __device__ void adjoint(const I& input, const O* adjOutput, AdjI& adjInput [, const param-type& param-name ]);
 * }
 * </code>
 * The input type provides a subscript operator <code>float operator[](size_t idx)</code>
 *  to fetch the input at the given index.
 * The output is any object with an <code>T operator[0](int idx)</code> and a typedef \c ValueType
 * specifying the expected datatype (float or half). Example: TensorAccessor or StaticArray
 */
class IEncoding
{
public:
    virtual ~IEncoding() = default;

    [[nodiscard]] virtual nlohmann::json toJson() const = 0;
    [[nodiscard]] virtual std::string id() const = 0;

    /**
     * The maximal addressed input channel, e.g. 2 for a 3D index.
     * Used to validate the network configuration
     */
    [[nodiscard]] virtual int maxInputChannel() const = 0;

    /**
     * The number of output channels
     */
    [[nodiscard]] virtual int numOutputChannels() const = 0;

    /**
     * The qualified name of the encoding as how it should be used
     * in the kernel code.
     * Can (and should) include all template specifications if needed.
     */
    [[nodiscard]] virtual std::string qualifiedName() const = 0;

    /**
     * Fills the stringstream with the code to include (and configure) this encoding
     */
    virtual void fillCode(std::stringstream& code) const = 0;

    /**
     * Returns true iff the encoding requires a parameter structure
     * as additional input.
     * This enables parameter training and all the '*parameter*' functions below.
     * The fully qualified name is obtained by \ref parameterName()
     */
    [[nodiscard]] virtual bool hasParameters() const { return false; }

    /**
     * Returns the fully-qualified name of the parameter structure.
     */
    [[nodiscard]] virtual std::string parameterName() const { return ""; }

    [[nodiscard]] virtual Tensor::Precision parameterPrecision(Tensor::Usage usage) const { return Tensor::FLOAT; }

    [[nodiscard]] virtual int parameterCount() const { return 0; }

    /**
     * Sets the underlying parameter to the specified tensor.
     * \param tensor the 1D tensor with \ref parameterCount() parameter of
     *   type parameterPrecision(Usage)
     * \param usage the usage for the parameters
     */
    virtual void setParameter(const Tensor& tensor, Tensor::Usage usage) {}

    /**
     * Writes the parameters into the constant field denoted by 'constantName'.
     * This field is then later passed to the evaluation kernel.
     * \see ckl::KernelFunction::fillConstantMemoryAsync
     */
    virtual void fillParameterConstant(
        const std::string& constantName, const ckl::KernelFunction& function, CUstream stream) {}

    /**
     * Zeros the gradients for the parameters of this encoding.
     * No-op if the encoding has no parameters
     */
    virtual void zeroGradients() {}


    /**
    * \brief Inference / forward pass through the activation function.
    *
    * \param input the input of shape (B, C), dtype=float32
    * \param output the output of shape (B, C), dtype=half16
    * \param stream the CUDA stream where the kernel is enqueued.
    */
    virtual void forward(const Tensor& input, Tensor& output, CUstream stream,
        const std::optional<const Tensor>& parametersForward);

    enum AdjointMode
    {
        INPUT_GRADIENTS = 0x1,
        PARAM_GRADIENTS = 0x2,
        ALL_GRADIENTS = INPUT_GRADIENTS | PARAM_GRADIENTS
    };

    /**
     * \brief Adjoint propagation through the activation function.
     * The tensors must be of half precision.

     * \param input the inputs of shape (B, C), dtype=float32
     * \param adjOutput the adjoint output of shape (B, C), dtype=half16
     * \param adjInput the adjoint input of shape (B, C), dtype=half16
     * \param stream the CUDA stream where the kernel is enqueued.
     */
    virtual void adjoint(const Tensor& input, const Tensor& adjOutput, Tensor& adjInput, CUstream stream,
        const std::optional<const Tensor>& parametersForward, const std::optional<const Tensor>& parametersGradients,
        int adjointMode = AdjointMode::ALL_GRADIENTS);

private:
    std::optional<ckl::KernelFunction> forwardKernel_;
    std::optional<ckl::KernelFunction> adjointKernel_[AdjointMode::ALL_GRADIENTS+1];
};
typedef std::shared_ptr<IEncoding> IEncoding_ptr;


/**
 * Specialization of IEncoding for volumetric encodings.
 * Volumetric Encodings can be included as child encodings in other models like EncodingLineIntegration.
 * To support this, a few extra methods need to be provided.
 *
 * The kernel code must provide a
 * <code>static constexpr int NumOutputs</code>
 * together with the functions specified in IEncoding.
 */
class IVolumetricEncoding : public IEncoding
{
public:
    // Number of dimensions (1D to 6D typically)
    [[nodiscard]] virtual int ndim() const = 0;

    typedef std::vector<float> BoundingBoxVector_t;

    /**
     * Returns the min point of the bounding box.
     * The returned vector has a length of \ref ndim().
     */
    [[nodiscard]] virtual BoundingBoxVector_t boundingBoxMin() const = 0;

    /**
     * Returns the side length of the bounding box.
     * The returned vector has a length of \ref ndim().
     */
    [[nodiscard]] virtual BoundingBoxVector_t boundingBoxSize() const = 0;

    /**
     * Returns the inverse side length of the bounding box.
     * The returned vector has a length of \ref ndim().
     */
    [[nodiscard]] virtual BoundingBoxVector_t boundingBoxInvSize() const = 0;

    /**
     * Fills the parameter data (see \ref fillParameterConstant()) into
     * the memory \c dst with maximal capacity of \c dstSize bytes.
     * The number of bytes written is returned
     */
    [[nodiscard]] virtual int fillParameterMemory(char* dst, int dstSize) = 0;
};


class EncodingFactory
{
private:
    EncodingFactory();
    typedef std::function<IEncoding_ptr(const nlohmann::json&)> factory_t;
    std::unordered_map<std::string, factory_t> encodings_;
public:
    static EncodingFactory& Instance();

    IEncoding_ptr create(const nlohmann::json& cfg);
};

QUICKMLP_NAMESPACE_END
