#ifndef _SCORCH_OPTIM_OPTIMIZER_H_
#define _SCORCH_OPTIM_OPTIMIZER_H_

/**
 * @file optimizer.h
 * @brief Base class for all parameter optimizers (equivalent to torch.optim.Optimizer).
 */

#include <memory>
#include <vector>

#include "tensor.h"

namespace scorch::optim {

/**
 * @brief Abstract base class for parameter optimizers.
 */
class Optimizer {
 public:
  explicit Optimizer(std::vector<std::shared_ptr<Tensor<float>>> parameters)
      : parameters_(std::move(parameters)) {}

  virtual ~Optimizer() = default;

  /**
   * @brief Performs a single optimization step (parameter update).
   */
  virtual void step() = 0;

  /**
   * @brief Resets the gradients of all optimized parameters to zero.
   */
  virtual void zero_grad() {
    for (auto& param : parameters_) {
      if (param) param->zero_grad();
    }
  }

  const std::vector<std::shared_ptr<Tensor<float>>>& parameters() const { return parameters_; }

 protected:
  std::vector<std::shared_ptr<Tensor<float>>> parameters_;
};

}  // namespace scorch::optim

#endif  // !_SCORCH_OPTIM_OPTIMIZER_H_
