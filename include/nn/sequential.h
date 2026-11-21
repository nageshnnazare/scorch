#ifndef _SCORCH_NN_SEQUENTIAL_H_
#define _SCORCH_NN_SEQUENTIAL_H_

/**
 * @file sequential.h
 * @brief Sequential container of modules (torch.nn.Sequential).
 */

#include <initializer_list>
#include <memory>
#include <vector>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief A sequential container: modules are evaluated in the order they were added.
 */
class Sequential : public Module {
 public:
  Sequential() = default;

  template <typename... Args>
    requires(sizeof...(Args) > 0 && (std::is_convertible_v<Args, std::shared_ptr<Module>> && ...))
  explicit Sequential(Args&&... args) {
    (add(std::forward<Args>(args)), ...);
  }

  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (auto& mod : modules) {
      add(mod);
    }
  }

  void add(std::shared_ptr<Module> module) {
    std::string name = std::to_string(modules_.size());
    modules_.push_back(register_module(std::move(name), std::move(module)));
  }

  void push_back(std::shared_ptr<Module> module) { add(std::move(module)); }

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override {
    auto current = input;
    for (auto& mod : modules_) {
      current = mod->forward(current);
    }
    return current;
  }

  std::size_t size() const { return modules_.size(); }
  bool empty() const { return modules_.empty(); }

  std::shared_ptr<Module> operator[](std::size_t index) const { return modules_.at(index); }

 private:
  std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_SEQUENTIAL_H_
