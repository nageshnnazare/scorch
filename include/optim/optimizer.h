#ifndef _SCORCH_OPTIM_OPTIMIZER_H_
#define _SCORCH_OPTIM_OPTIMIZER_H_

/**
 * @file optimizer.h
 * @brief Base class for all first-order parameter optimizers in scorch.
 *
 * @details
 * `scorch::optim::Optimizer` provides the unified interface for optimizing
 * network parameters using accumulated autograd gradients, directly mirroring
 * `torch.optim.Optimizer` in PyTorch.
 *
 * Typical optimization loop:
 * @code{.cpp}
 * scorch::optim::SGD optimizer(model->parameters(), 0.01f);
 *
 * for (int epoch = 0; epoch < epochs; ++epoch) {
 *   optimizer.zero_grad();
 *   auto loss = criterion->forward(model->forward(x), y);
 *   loss->backward();
 *   optimizer.step();
 * }
 * @endcode
 *
 * @ingroup optim
 */

#include <memory>
#include <utility>
#include <vector>

#include "tensor.h"

namespace scorch::optim {

/**
 * @brief Abstract base class for parameter optimizers.
 *
 * Holds references to trainable parameter tensors and defines pure-virtual
 * step() for algorithm-specific weight update rules.
 */
class Optimizer {
 public:
  /**
   * @brief Construct an optimizer targeting a collection of parameters.
   * @param parameters List of shared pointers to trainable `Tensor<float>` parameters.
   */
  explicit Optimizer(std::vector<std::shared_ptr<Tensor<float>>> parameters)
      : parameters_(std::move(parameters)) {}

  virtual ~Optimizer() = default;

  /**
   * @brief Performs a single parameter update step using current gradients.
   *
   * Subclasses override this method to implement concrete optimization algorithms
   * such as SGD, Adam, RMSprop, etc.
   */
  virtual void step() = 0;

  /**
   * @brief Resets the gradients of all managed parameters to zero.
   *
   * Should be called at the start of each training iteration before the forward
   * and backward passes to prevent gradient accumulation across iterations.
   */
  virtual void zero_grad() {
    for (auto& param : parameters_) {
      if (param) param->zero_grad();
    }
  }

  /**
   * @brief Access the list of parameter tensors managed by this optimizer.
   * @return Const reference to the vector of parameter shared pointers.
   */
  const std::vector<std::shared_ptr<Tensor<float>>>& parameters() const { return parameters_; }

 protected:
  /// List of parameter tensors tracked and updated by the optimizer.
  std::vector<std::shared_ptr<Tensor<float>>> parameters_;
};

}  // namespace scorch::optim

#endif  // !_SCORCH_OPTIM_OPTIMIZER_H_
