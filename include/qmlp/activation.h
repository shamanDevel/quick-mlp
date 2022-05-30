#pragma once

#include "common.h"
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "ckl/kernel_loader.h"

QUICKMLP_NAMESPACE_BEGIN
    /**
     * Defines an activation function.
     * An activation function defines two functions, acting point-wise:
     *
     * Forward function (inference):
     *  - Input: <code>half x</code>
     *  - Output: <code>half z</code>
     *  - Example for ReLU:
     *    <code> z = __hmax(x, 0)</code>
     *
     * Adjoint function (for training)
     *  - Input: <code>half x, adjz</code> the input for the forward function and the adjoint of the output
     *  - Output: <code>half adjx</code> the adjoint of the input.
     *  - Example for ReLU:
     *    <code> adjx = x>0 ? adjx : 0</code>
     */
class Activation
{
    const std::string id_;
    const std::string forward_;
    const std::string adjoint_;
    const std::string code_;

public:
    Activation(const std::string& id, const std::string& forward, const std::string& adjoint);

    Activation(const nlohmann::json& cfg);

    [[nodiscard]] nlohmann::json toJson() const;

    [[nodiscard]] const std::string& id() const
    {
        return id_;
    }

    [[nodiscard]] const std::string& forward() const
    {
        return forward_;
    }

    [[nodiscard]] const std::string& adjoint() const
    {
        return adjoint_;
    }

    /**
     * The code definition to be included in the code generation.
     * Defines the following snippet:
     * <code>
     * struct {id()}
     * {
     *    static half forward(half x);
     *    static half adjoint(half x, half adjz);
     * }
     */
    [[nodiscard]] std::string code() const
    {
        return code_;
    }
};
typedef std::shared_ptr<Activation> Activation_ptr;

/**
 * Loader for activations from Json specifications
 */
class ActivationFactory
{
public:
    /**
     * Loads the activations from an activation specification tag.
     * 'cfg' is an array with filenames, each file then contains
     * an array of activation specifications.
     *
     * Search order for the files:
     * - Direct (relative path to the current directory or an absolute path)
     * - Relative to 'parent'
     * - From the kernel loader 'loader' to include built-in activations
     *
     * \param cfg the json array with a list of filenames to collect the activations from.
     * \param parent the parent folder where the configuration was saved.
     *   Used to resolve activation specifications relative to the network config file
     * \parma loader the kernel loader for the built-in activations
     */
    ActivationFactory(const nlohmann::json& cfg, 
        const std::filesystem::path& parent, ckl::KernelLoader_ptr loader);

    /**
     * Returns the activation with the given identifier.
     * Throws an exception if no such key was found
     */
    Activation_ptr get(const std::string& key) const;

private:
    void parseFile(const std::filesystem::path& file);
    void parseFile(const nlohmann::json& j);
    void parseActivation(const nlohmann::json& cfg);

    std::unordered_map<std::string, Activation_ptr> activations_;
};
typedef std::shared_ptr<ActivationFactory> ActivationFactory_ptr;

QUICKMLP_NAMESPACE_END
