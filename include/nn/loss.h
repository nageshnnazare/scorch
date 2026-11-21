#ifndef _SCORCH_NN_LOSS_H_
#define _SCORCH_NN_LOSS_H_

/**
 * @file loss.h
 * @brief Loss function modules for supervised learning.
 *
 * @details
 * Loss modules derive from `scorch::nn::Module` and expose a two-argument
 * `forward(prediction, target)` overload alongside the mandatory single-
 * argument override (which throws to prevent misuse).
 *
 * | Class             | Use-case                              | PyTorch equiv              |
 * |-------------------|---------------------------------------|----------------------------|
 * | MSELoss           | Regression / continuous targets        | `nn.MSELoss()`             |
 * | CrossEntropyLoss  | Multi-class classification (logits)   | `nn.CrossEntropyLoss()`    |
 *
 * @ingroup nn
 */

#include <memory>

#include "nn/module.h"

namespace scorch::nn {

// =============================================================================
// MSELoss
// =============================================================================

/**
 * @brief Mean Squared Error loss between predictions and targets.
 *
 * @details
 * Computes the mean of squared element-wise differences:
 *
 * \f[
 *   \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
 * \f]
 *
 * where \f$y_i\f$ is the prediction and \f$\hat{y}_i\f$ is the target.
 *
 * MSE penalises large errors heavily and is differentiable everywhere, making
 * it the standard loss for regression tasks.
 *
 * **Backpropagation:** gradient w.r.t. prediction is
 * \f$\frac{2}{N}(\mathbf{y} - \hat{\mathbf{y}})\f$.
 *
 * ### Usage
 * @code{.cpp}
 * auto criterion = std::make_shared<scorch::nn::MSELoss>();
 *
 * auto pred   = model->forward(x);         // shape {N, 1}
 * auto target = scorch::from({N, 1}, ...); // shape {N, 1}
 * auto loss   = criterion->forward(pred, target);
 *
 * loss->backward();
 * @endcode
 *
 * @note This class does **not** support the single-argument `forward(input)`
 *       signature inherited from `Module`.  Calling it throws immediately.
 */
class MSELoss : public Module {
 public:
  /**
   * @brief Not supported — always throws.
   *
   * @details
   * MSELoss requires both a prediction and a target tensor.  This override
   * exists only to satisfy the pure-virtual contract of `Module`.
   *
   * @throws std::invalid_argument Always.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  /**
   * @brief Compute MSE loss between `prediction` and `target`.
   *
   * @details
   * Both tensors must have identical shapes.  The result is a scalar tensor
   * (shape `{1}`) with `requires_grad` automatically set if either operand
   * has gradients enabled.
   *
   * @param prediction  Model output tensor; `requires_grad` should be `true`.
   * @param target      Ground-truth tensor (usually detached / no grad).
   * @return            Scalar loss tensor.
   *
   * @throws std::invalid_argument If the shapes of `prediction` and `target`
   *         differ.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& prediction,
                                         const std::shared_ptr<Tensor<float>>& target);
};

// =============================================================================
// CrossEntropyLoss
// =============================================================================

/**
 * @brief Cross-entropy loss combining log-softmax and negative log-likelihood.
 *
 * @details
 * Accepts **raw logits** (pre-softmax scores) and **integer class indices**
 * as targets, then computes:
 *
 * \f[
 *   \mathcal{L}_{\text{CE}} =
 *     -\frac{1}{N} \sum_{i=1}^{N}
 *       \log \frac{e^{z_{i,c_i}}}{\sum_{j} e^{z_{i,j}}}
 * \f]
 *
 * where \f$z_i\f$ is the logit vector for sample \f$i\f$ and \f$c_i\f$ is
 * its true class index.
 *
 * **Numerical stability:** log-softmax is computed with the log-sum-exp trick
 * (subtracting \f$\max z_i\f$) to avoid overflow in the exponent.
 *
 * **This fused form is preferred** over applying `Softmax` and then computing
 * log manually, as the fused gradient is more numerically stable.
 *
 * **Backpropagation:**
 * \f[
 *   \frac{\partial \mathcal{L}}{\partial z_{i,j}} =
 *     \frac{1}{N}(\text{softmax}(z_i)_j - \mathbf{1}[j = c_i])
 * \f]
 *
 * ### Usage
 * @code{.cpp}
 * // logits: {N, C} float tensor (raw class scores, no softmax)
 * // targets: {N} int tensor   (class indices 0 … C-1)
 * auto criterion = std::make_shared<scorch::nn::CrossEntropyLoss>();
 *
 * auto logits  = model->forward(x);              // {N, C}
 * auto targets = scorch::from<int>({N}, labels); // {N}
 * auto loss    = criterion->forward(logits, targets);
 *
 * loss->backward();
 * @endcode
 *
 * @note This class does **not** support the single-argument `forward(input)`
 *       signature inherited from `Module`.  Calling it throws immediately.
 */
class CrossEntropyLoss : public Module {
 public:
  /**
   * @brief Not supported — always throws.
   *
   * @details
   * CrossEntropyLoss requires both logits and integer targets.  This override
   * exists only to satisfy the pure-virtual contract of `Module`.
   *
   * @throws std::invalid_argument Always.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  /**
   * @brief Compute cross-entropy loss from logits and class indices.
   *
   * @param logits   Float tensor of shape `{N, C}` — raw (pre-softmax) class
   *                 scores.  `requires_grad` should be `true`.
   * @param targets  Integer tensor of shape `{N}` — ground-truth class indices
   *                 in the range \f$[0, C)\f$.
   * @return         Scalar loss tensor (shape `{1}`).
   *
   * @throws std::invalid_argument If the batch sizes of `logits` and `targets`
   *         do not match, or if any target index is out of range.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& logits,
                                         const std::shared_ptr<Tensor<int>>& targets);
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_LOSS_H_
