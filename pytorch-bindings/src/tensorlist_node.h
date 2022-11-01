#pragma once

#include <torch/torch.h>


// Context to save information during forward that can be accessed in backward
struct TensorlistAutogradContext
{
    TensorlistAutogradContext() = default;
    TensorlistAutogradContext(const TensorlistAutogradContext& other) = delete;
    TensorlistAutogradContext& operator=(const TensorlistAutogradContext& other) = delete;

    // Can be used to save non-variable data for backward()
    ska::flat_hash_map<std::string, at::IValue> saved_data;

    // Saves the list of variables for a future call to backward(). This
    // should be called at most once from inside of forward().
    void save_for_backward(torch::autograd::variable_list to_save);
    // Marks variables in the list as modified in an in-place operation. This
    // should be called at most once from inside of forward() and all arguments
    // should be inputs.
    void mark_dirty(const torch::autograd::variable_list& inputs);
    // Marks outputs in the list as not requiring gradients. This should be called
    // at most once from inside of forward() and all arguments should be outputs.
    void mark_non_differentiable(const torch::autograd::variable_list& outputs);

    // Get the list of variables that were saved in forward using
    // save_for_backward(). Before returning them to the user, a check is made to
    // ensure that they were not modified by any in-place operations.
    torch::autograd::variable_list get_saved_variables() const;
    const std::unordered_set<at::TensorImpl*>& get_dirty() const;
    const std::unordered_set<at::TensorImpl*>& get_non_differentiable() const;

private:
    std::unordered_set<at::TensorImpl*> non_differentiable_;
    std::unordered_set<at::TensorImpl*> dirty_inputs_;
    std::vector<torch::autograd::SavedVariable> saved_variables_;
    torch::autograd::variable_list to_save_;

    // The CppNode in the autograd graph that owns this AutogradContext. We need a
    // weak_ptr to avoid a refcycle. Since grad_fn_ owns this AutogradContext, it
    // will always be alive when we want to use it.
    std::weak_ptr<torch::autograd::Node> grad_fn_;
    bool has_freed_buffers_;

    void save_variables();

    friend struct TensorlistNode;
};

/**
 * The main entry point
 */
struct TensorlistFunction
{
    static torch::autograd::variable_list apply(
        torch::autograd::variable_list args,
        std::function<torch::autograd::variable_list(TensorlistAutogradContext*, torch::autograd::variable_list)> forward,
        std::function<torch::autograd::variable_list(TensorlistAutogradContext*, torch::autograd::variable_list)> backward);
};

struct TensorlistNode : public torch::autograd::Node
{

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;
    TensorlistAutogradContext ctx_;
    std::vector<bool> is_variable_input_;
    std::vector<torch::autograd::VariableInfo> input_info_;
    std::vector<torch::autograd::VariableInfo> output_info_;
    std::function<torch::autograd::variable_list(TensorlistAutogradContext*, torch::autograd::variable_list)> backward_;

    void release_variables() override;

    void set_ctx_grad_fn(const std::shared_ptr<Node>& node);
    void save_variables_to_ctx();
};