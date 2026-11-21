#ifndef _SCORCH_NN_SEQUENTIAL_H_
#define _SCORCH_NN_SEQUENTIAL_H_

/**
 * @file sequential.h
 * @brief Ordered container that chains modules end-to-end.
 *
 * @details
 * `scorch::nn::Sequential` holds an ordered list of modules and feeds the
 * output of each module as the input to the next.  It is equivalent to
 * `torch.nn.Sequential` in PyTorch.
 *
 * ### Construction
 * Modules can be added at construction time (variadic or initialiser-list)
 * or dynamically via `add()` / `push_back()`:
 *
 * @code{.cpp}
 * // Variadic construction (most common)
 * auto net = std::make_shared<scorch::nn::Sequential>(
 *     std::make_shared<scorch::nn::Linear>(784, 256),
 *     std::make_shared<scorch::nn::ReLU>(),
 *     std::make_shared<scorch::nn::Linear>(256, 10));
 *
 * // Dynamic construction
 * scorch::nn::Sequential net;
 * net.add(std::make_shared<scorch::nn::Linear>(784, 256));
 * net.add(std::make_shared<scorch::nn::ReLU>());
 * @endcode
 *
 * ### Naming convention
 * Modules are auto-named `"0"`, `"1"`, `"2"`, … (their index in the list),
 * matching PyTorch's default naming for `Sequential`.  These names appear in
 * `named_parameters()` output.
 *
 * @ingroup nn
 */

#include <initializer_list>
#include <memory>
#include <vector>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Chains an ordered sequence of modules; output of one feeds the next.
 *
 * @details
 * The forward pass simply iterates the module list and passes the running
 * tensor through each one in order:
 *
 * @code
 * output = modules[n-1]( … modules[1]( modules[0](input) ) … )
 * @endcode
 *
 * All child modules are registered (via `register_module`) so that
 * `parameters()`, `train()`, and `eval()` propagate automatically.
 */
class Sequential : public Module {
 public:
  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  /// @brief Default constructor — creates an empty container.
  Sequential() = default;

  /**
   * @brief Variadic constructor — add any number of modules at once.
   *
   * @details
   * Each argument must be (or be implicitly convertible to)
   * `std::shared_ptr<Module>`.  Modules are stored in the order they are
   * passed.
   *
   * @code{.cpp}
   * auto net = std::make_shared<Sequential>(
   *     std::make_shared<Linear>(2, 4),
   *     std::make_shared<ReLU>(),
   *     std::make_shared<Linear>(4, 1));
   * @endcode
   *
   * @tparam Args  Pack of types convertible to `std::shared_ptr<Module>`.
   * @param  args  Module shared pointers to add.
   */
  template <typename... Args>
    requires(sizeof...(Args) > 0 && (std::is_convertible_v<Args, std::shared_ptr<Module>> && ...))
  explicit Sequential(Args&&... args) {
    (add(std::forward<Args>(args)), ...);
  }

  /**
   * @brief Initialiser-list constructor for brace-initialised module lists.
   *
   * @code{.cpp}
   * Sequential net({
   *     std::make_shared<Linear>(2, 4),
   *     std::make_shared<ReLU>()});
   * @endcode
   *
   * @param modules  Brace-enclosed list of `shared_ptr<Module>`.
   */
  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (auto& mod : modules) {
      add(mod);
    }
  }

  // ---------------------------------------------------------------------------
  // Dynamic modification
  // ---------------------------------------------------------------------------

  /**
   * @brief Append a module to the end of the sequence.
   *
   * @details
   * The module is registered under the name equal to its zero-based index
   * in the list (e.g. the third module added is named `"2"`).
   *
   * @param module  Module to append.  Must not be `nullptr`.
   */
  void add(std::shared_ptr<Module> module) {
    std::string name = std::to_string(modules_.size());
    modules_.push_back(register_module(std::move(name), std::move(module)));
  }

  /**
   * @brief Alias for `add()` — matches the STL container naming convention.
   * @param module  Module to append.
   */
  void push_back(std::shared_ptr<Module> module) { add(std::move(module)); }

  // ---------------------------------------------------------------------------
  // Forward pass
  // ---------------------------------------------------------------------------

  /**
   * @brief Pass `input` sequentially through each module.
   *
   * @details
   * Iterates the internal module list in insertion order.  The output of
   * module \f$i\f$ is passed as the input to module \f$i+1\f$.
   *
   * @param input  Input tensor fed to the first module.
   * @return       Output tensor from the last module.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override {
    auto current = input;
    for (auto& mod : modules_) {
      current = mod->forward(current);
    }
    return current;
  }

  // ---------------------------------------------------------------------------
  // Inspection
  // ---------------------------------------------------------------------------

  /// @brief Number of modules currently in the container.
  std::size_t size() const { return modules_.size(); }

  /// @brief `true` if no modules have been added yet.
  bool empty() const { return modules_.empty(); }

  /**
   * @brief Access a module by its zero-based index.
   * @param index  Position in the sequence (0-based).
   * @return       Shared pointer to the module at `index`.
   * @throws std::out_of_range If `index >= size()`.
   */
  std::shared_ptr<Module> operator[](std::size_t index) const { return modules_.at(index); }

 private:
  /// Ordered list of registered child modules.
  std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_SEQUENTIAL_H_
