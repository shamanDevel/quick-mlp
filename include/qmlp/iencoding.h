#pragma once

#include "common.h"

#include <vector>
#include <sstream>
#include <unordered_map>
#include <functional>
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
 *     template<typename I>
 *     static __device__ void forward(const I input, half* output [, const param-type& param-name ]);
 *
 *     template<bool EvaluateInputGradients, bool EvaluateParameterGradients, typename I, typename O>
 *     static __device__ void adjoint(const I& input, const half* adjOutput, O& adjInput [, const param-type& param-name ]);
 * }
 * </code>
 * The input type provides a subscript operator <code>float operator[](size_t idx)</code>
 *  to fetch the input at the given index.
 * The pointer to the output 
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

    [[nodiscard]] virtual size_t parameterCount() const { return 0; }

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

};
typedef std::shared_ptr<IEncoding> IEncoding_ptr;

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
