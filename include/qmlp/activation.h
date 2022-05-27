#pragma once

#include "common.h"
#include <string>
#include <nlohmann/json.hpp>

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

QUICKMLP_NAMESPACE_END
