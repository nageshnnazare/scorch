#ifndef _SCORCH_NN_MODULE_H_
#define _SCORCH_NN_MODULE_H_

/**
 * @file module.h
 * @brief Base class for all neural network modules (equivalent to torch.nn.Module).
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"

namespace scorch::nn {

/**
 * @brief Base class for all neural network modules in scorch.
 *
 * Modules can contain trainable parameters (Tensors with requires_grad=true)
 * as well as nested submodules.
 */
class Module : public std::enable_shared_from_this<Module> {
 public:
  virtual ~Module() = default;

  /**
   * @brief Perform the forward computation of the module.
   * @param input Input tensor.
   * @return Output tensor resulting from module evaluation.
   */
  virtual std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) = 0;

  /**
   * @brief Functional call operator forwarding to forward().
   */
  std::shared_ptr<Tensor<float>> operator()(const std::shared_ptr<Tensor<float>>& input) {
    return forward(input);
  }

  /**
   * @brief Set module and all submodules into training mode.
   */
  virtual void train(bool mode = true) {
    training_ = mode;
    for (auto& [name, submod] : submodules_) {
      if (submod) submod->train(mode);
    }
  }

  /**
   * @brief Set module and all submodules into evaluation mode.
   */
  virtual void eval() { train(false); }

  /**
   * @brief Check if the module is in training mode.
   */
  bool is_training() const { return training_; }

  /**
   * @brief Collect all parameters in this module and its submodules recursively.
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
   * @brief Collect all named parameters recursively.
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

  /**
   * @brief Zero the gradients of all parameters.
   */
  void zero_grad() {
    for (auto& param : parameters()) {
      if (param) param->zero_grad();
    }
  }

 protected:
  /**
   * @brief Register a parameter with this module.
   */
  std::shared_ptr<Tensor<float>>& register_parameter(std::string name,
                                                     std::shared_ptr<Tensor<float>> param) {
    parameters_.emplace_back(std::move(name), param);
    return parameters_.back().second;
  }

  /**
   * @brief Register a nested child submodule.
   */
  template <typename TModule>
  std::shared_ptr<TModule> register_module(std::string name, std::shared_ptr<TModule> module) {
    submodules_.emplace_back(std::move(name), module);
    return module;
  }

 private:
  bool training_{true};
  std::vector<std::pair<std::string, std::shared_ptr<Tensor<float>>>> parameters_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_MODULE_H_
