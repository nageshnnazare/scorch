#ifndef _SCORCH_NN_MODULE_H_
#define _SCORCH_NN_MODULE_H_

/**
 * @file module.h
 * @brief Base class for all neural network modules in scorch.
 *
 * @details
 * `scorch::nn::Module` is the foundational building block of every neural
 * network constructed with this library — analogous to `torch.nn.Module` in
 * PyTorch.
 *
 * Every learnable layer (e.g. `Linear`), activation function wrapper (e.g.
 * `ReLU`), loss function (e.g. `MSELoss`), or composite model must inherit
 * from `Module` and implement its `forward()` pure-virtual method.
 *
 * ### Lifecycle
 * ```
 * construct → register_parameter / register_module
 *           → forward()   (called via operator() or forward())
 *           → zero_grad() → backward() → optimizer step
 * ```
 *
 * ### Training vs Evaluation mode
 * Modules start in **training** mode.  Call `eval()` before inference to
 * disable stochastic behaviour (e.g. Dropout).  `train()` switches back.
 * Both calls propagate recursively to all registered submodules.
 *
 * ### Parameters
 * Parameters are `Tensor<float>` instances registered via
 * `register_parameter()`.  `parameters()` returns a flat list suitable for
 * passing directly to an optimiser:
 *
 * ```cpp
 * auto model = std::make_shared<scorch::nn::Linear>(784, 10);
 * scorch::optim::SGD optim(model->parameters(), 0.01f);
 * ```
 *
 * @ingroup nn
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"

namespace scorch::nn {

/**
 * @brief Abstract base class for all neural network modules.
 *
 * @details
 * Derive from `Module` to build trainable layers.  At minimum, override
 * `forward()`.  Trainable weights should be registered with
 * `register_parameter()` so that `parameters()` and `zero_grad()` pick them
 * up automatically.  Child sub-networks should be registered with
 * `register_module()` so that `train()` / `eval()` and `parameters()`
 * propagate recursively.
 *
 * Example — a single linear + ReLU block:
 * @code{.cpp}
 * class MyBlock : public scorch::nn::Module {
 *  public:
 *   MyBlock(int in, int out)
 *       : fc_(register_module("fc", std::make_shared<scorch::nn::Linear>(in, out))) {}
 *
 *   std::shared_ptr<Tensor<float>> forward(
 *       const std::shared_ptr<Tensor<float>>& x) override {
 *     return scorch::relu(fc_->forward(x));
 *   }
 *
 *  private:
 *   std::shared_ptr<scorch::nn::Linear> fc_;
 * };
 * @endcode
 */
class Module : public std::enable_shared_from_this<Module> {
 public:
  virtual ~Module() = default;

  // ---------------------------------------------------------------------------
  // Core interface
  // ---------------------------------------------------------------------------

  /**
   * @brief Perform the forward pass of the module.
   *
   * @details
   * All derived classes **must** override this method.  It should compute the
   * module's output from `input`, building the autograd computation graph by
   * using scorch tensor operations (which record grad-functions automatically
   * when any operand has `requires_grad == true`).
   *
   * @param input  Input tensor (batch dimension first by convention).
   * @return       Output tensor; may share graph nodes with `input`.
   */
  virtual std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) = 0;

  /**
   * @brief Convenience call operator that delegates to `forward()`.
   *
   * @details
   * Allows modules to be called like functions, mirroring PyTorch syntax:
   * @code{.cpp}
   * auto y = model(x);   // equivalent to model->forward(x)
   * @endcode
   *
   * @param input  Input tensor.
   * @return       Result of `forward(input)`.
   */
  std::shared_ptr<Tensor<float>> operator()(const std::shared_ptr<Tensor<float>>& input) {
    return forward(input);
  }

  // ---------------------------------------------------------------------------
  // Training / evaluation mode
  // ---------------------------------------------------------------------------

  /**
   * @brief Switch the module (and all submodules) into training or eval mode.
   *
   * @details
   * When `mode == true` the module behaves stochastically (e.g. Dropout
   * drops activations).  When `mode == false` stochastic behaviour is
   * suppressed and the module acts deterministically.
   *
   * The flag is propagated recursively to every registered submodule.
   *
   * @param mode  `true` → training mode (default), `false` → eval mode.
   */
  virtual void train(bool mode = true) {
    training_ = mode;
    for (auto& [name, submod] : submodules_) {
      if (submod) submod->train(mode);
    }
  }

  /**
   * @brief Switch the module (and all submodules) into evaluation mode.
   *
   * @details
   * Shorthand for `train(false)`.  Use before running inference to disable
   * layers whose behaviour differs between training and evaluation (e.g.
   * Dropout, BatchNorm).
   */
  virtual void eval() { train(false); }

  /**
   * @brief Query whether the module is currently in training mode.
   * @return `true` if training, `false` if eval.
   */
  bool is_training() const { return training_; }

  // ---------------------------------------------------------------------------
  // Parameter inspection
  // ---------------------------------------------------------------------------

  /**
   * @brief Return all trainable parameters in this module and its submodules.
   *
   * @details
   * Parameters are collected in registration order: own parameters first,
   * then each submodule's parameters recursively in registration order.
   *
   * The returned vector is typically passed directly to an optimiser:
   * @code{.cpp}
   * scorch::optim::Adam opt(model->parameters(), 1e-3f);
   * @endcode
   *
   * @return Flat vector of shared_ptr to every registered parameter.
   */
  virtual std::vector<std::shared_ptr<Tensor<float>>> parameters() const {
    std::vector<std::shared_ptr<Tensor<float>>> params;
    for (const auto& [name, param] : parameters_) {
      if (param) params.push_back(param);
    }
    for (const auto& [name, submod] : submodules_) {
      if (submod) {
        auto sub_params = submod->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
      }
    }
    return params;
  }

  /**
   * @brief Return all parameters paired with their dot-separated names.
   *
   * @details
   * Names follow the same dotted-path convention as PyTorch:
   * - Own parameter `"weight"` in module `"fc"` → `"fc.weight"`.
   * - Nested module `"block"` with parameter `"weight"` → `"block.weight"`.
   *
   * Useful for inspecting which tensors belong to which sub-graph, and for
   * targeted learning-rate schedules.
   *
   * @param prefix  Dot-separated name prefix prepended to each parameter name.
   *                Defaults to `""` (top-level call).
   * @return        Vector of `(name, tensor)` pairs in registration order.
   */
  virtual std::vector<std::pair<std::string, std::shared_ptr<Tensor<float>>>> named_parameters(
      const std::string& prefix = "") const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor<float>>>> named_params;
    for (const auto& [name, param] : parameters_) {
      if (param) {
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        named_params.emplace_back(full_name, param);
      }
    }
    for (const auto& [name, submod] : submodules_) {
      if (submod) {
        std::string full_prefix = prefix.empty() ? name : prefix + "." + name;
        auto sub_named = submod->named_parameters(full_prefix);
        named_params.insert(named_params.end(), sub_named.begin(), sub_named.end());
      }
    }
    return named_params;
  }

  // ---------------------------------------------------------------------------
  // Gradient utilities
  // ---------------------------------------------------------------------------

  /**
   * @brief Set the gradient of every parameter to zero.
   *
   * @details
   * Must be called before each backward pass to prevent gradient accumulation
   * across iterations.  Internally calls `Tensor::zero_grad()` on each
   * parameter returned by `parameters()`.
   *
   * Typical training loop:
   * @code{.cpp}
   * model->zero_grad();
   * auto loss = criterion->forward(model->forward(x), y);
   * loss->backward();
   * optimizer.step();
   * @endcode
   */
  void zero_grad() {
    for (auto& param : parameters()) {
      if (param) param->zero_grad();
    }
  }

 protected:
  // ---------------------------------------------------------------------------
  // Registration helpers (called from derived class constructors)
  // ---------------------------------------------------------------------------

  /**
   * @brief Register a tensor as a named learnable parameter of this module.
   *
   * @details
   * The tensor is stored internally and returned (and exposed) for direct
   * member access.  It will appear in `parameters()` and `named_parameters()`.
   *
   * The tensor should have `requires_grad == true` so that gradients are
   * tracked.  Example:
   * @code{.cpp}
   * weight = register_parameter(
   *     "weight", scorch::rand({in, out}, -k, k, true));
   * @endcode
   *
   * @param name   Identifier used in `named_parameters()`.
   * @param param  Tensor to register (should have `requires_grad = true`).
   * @return       Reference to the stored shared_ptr for direct member binding.
   */
  std::shared_ptr<Tensor<float>>& register_parameter(std::string name,
                                                     std::shared_ptr<Tensor<float>> param) {
    parameters_.emplace_back(std::move(name), param);
    return parameters_.back().second;
  }

  /**
   * @brief Register a child module as a named submodule.
   *
   * @details
   * The submodule participates in recursive `parameters()`, `train()`, and
   * `eval()` calls.  Example:
   * @code{.cpp}
   * fc_ = register_module("fc", std::make_shared<Linear>(64, 32));
   * @endcode
   *
   * @tparam TModule  Concrete module type (must derive from `Module`).
   * @param  name     Identifier used in `named_parameters()` prefix.
   * @param  module   Shared pointer to the child module.
   * @return          The same shared_ptr (for member binding convenience).
   */
  template <typename TModule>
  std::shared_ptr<TModule> register_module(std::string name, std::shared_ptr<TModule> module) {
    submodules_.emplace_back(std::move(name), module);
    return module;
  }

 private:
  /// Whether the module is in training mode (true) or eval mode (false).
  bool training_{true};

  /// Ordered list of (name, tensor) pairs for this module's own parameters.
  std::vector<std::pair<std::string, std::shared_ptr<Tensor<float>>>> parameters_;

  /// Ordered list of (name, module) pairs for registered child submodules.
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_MODULE_H_
