/**
 * @file loss.cc
 * @brief Implementations of MSELoss and CrossEntropyLoss modules.
 *
 * @details
 * Both loss classes wrap the corresponding free-function ops in
 * `scorch::ops::loss`.  The single-argument `forward()` override required by
 * `Module`'s pure-virtual interface throws immediately to prevent silent
 * misuse — always call the two-argument overload.
 *
 * ### MSELoss
 * Delegates to `scorch::mse_loss(prediction, target)`, which computes:
 * \f[
 *   \mathcal{L} = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2
 * \f]
 * and registers a backward function that computes
 * \f$\frac{2}{N}(\mathbf{y} - \hat{\mathbf{y}})\f$.
 *
 * ### CrossEntropyLoss
 * Delegates to `scorch::cross_entropy_loss(logits, targets)`, which fuses
 * log-softmax + negative log-likelihood using the numerically stable
 * log-sum-exp trick:
 * \f[
 *   \mathcal{L} = -\frac{1}{N} \sum_i
 *     \left(z_{i,c_i} - \log\sum_j e^{z_{i,j}}\right)
 * \f]
 */

#include "nn/loss.h"

#include <stdexcept>

#include "ops/loss.h"

namespace scorch::nn {

// -----------------------------------------------------------------------------
// MSELoss
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> MSELoss::forward(const std::shared_ptr<Tensor<float>>& /*input*/) {
  // Guard: the Module interface requires a one-argument forward(), but MSELoss
  // cannot compute a meaningful loss without a target tensor.
  throw std::invalid_argument(
      "MSELoss::forward(input) is not supported.  "
      "Use forward(prediction, target) instead.");
}

std::shared_ptr<Tensor<float>> MSELoss::forward(const std::shared_ptr<Tensor<float>>& prediction,
                                                const std::shared_ptr<Tensor<float>>& target) {
  // Delegate to the autograd-aware free function in ops/loss.
  // The result is a scalar tensor; calling .backward() on it propagates
  // dL/d(prediction) = 2/N * (prediction - target).
  return scorch::mse_loss(prediction, target);
}

// -----------------------------------------------------------------------------
// CrossEntropyLoss
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> CrossEntropyLoss::forward(
    const std::shared_ptr<Tensor<float>>& /*input*/) {
  // Guard: CrossEntropyLoss needs both logits and integer class targets.
  throw std::invalid_argument(
      "CrossEntropyLoss::forward(input) is not supported.  "
      "Use forward(logits, targets) instead.");
}

std::shared_ptr<Tensor<float>> CrossEntropyLoss::forward(
    const std::shared_ptr<Tensor<float>>& logits, const std::shared_ptr<Tensor<int>>& targets) {
  // Delegate to the numerically stable fused log-softmax + NLL implementation.
  // Backward: dL/dz_{i,j} = (softmax(z_i)_j - 1[j == c_i]) / N.
  return scorch::cross_entropy_loss(logits, targets);
}

}  // namespace scorch::nn
