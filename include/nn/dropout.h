#ifndef _SCORCH_NN_DROPOUT_H_
#define _SCORCH_NN_DROPOUT_H_

/**
 * @file dropout.h
 * @brief Dropout regularisation module.
 *
 * @details
 * Dropout randomly zeroes a fraction of elements in the input tensor during
 * training, forcing the network to learn redundant representations and
 * reducing co-adaptation of neurons (a form of ensemble regularisation).
 *
 * This implementation uses **inverted dropout**: surviving activations are
 * scaled up by \f$\frac{1}{1-p}\f$ so that the expected sum of activations
 * remains the same at both train and test time.  During evaluation the module
 * acts as the identity function (no scaling, no masking).
 *
 * @note This is equivalent to `torch.nn.Dropout` in PyTorch.
 *
 * @ingroup nn
 */

#include <memory>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Randomly zeroes elements with probability `p` during training.
 *
 * @details
 * During **training**:
 * \f[
 *   y_i =
 *   \begin{cases}
 *     \frac{x_i}{1 - p} & \text{with probability } 1-p \\
 *     0                 & \text{with probability } p
 *   \end{cases}
 * \f]
 *
 * During **evaluation** (`eval()` mode): \f$y = x\f$ (identity, no-op).
 *
 * ### Inverted dropout rationale
 * Scaling by \f$\frac{1}{1-p}\f$ at training time means test-time inference
 * does **not** need to rescale outputs — the expected activation magnitude is
 * preserved automatically.
 *
 * ### Backpropagation
 * The binary mask is multiplied element-wise with the input, so the backward
 * pass propagates gradients only through surviving (non-zero) elements.
 *
 * ### Usage
 * @code{.cpp}
 * auto net = std::make_shared<scorch::nn::Sequential>(
 *     std::make_shared<scorch::nn::Linear>(512, 256),
 *     std::make_shared<scorch::nn::ReLU>(),
 *     std::make_shared<scorch::nn::Dropout>(0.5f),   // 50% drop rate
 *     std::make_shared<scorch::nn::Linear>(256, 10));
 *
 * net->train();         // activates dropout
 * auto y_train = net->forward(x);
 *
 * net->eval();          // disables dropout for inference
 * auto y_test = net->forward(x);
 * @endcode
 */
class Dropout : public Module {
 public:
  /**
   * @brief Construct a Dropout module.
   *
   * @param p  Probability of an element being zeroed.  Must satisfy
   *           \f$0 \leq p < 1\f$.  Default: `0.5`.
   *
   * @throws std::invalid_argument If `p < 0` or `p >= 1`.
   */
  explicit Dropout(float p = 0.5f);

  /**
   * @brief Apply dropout to `input` (or return it unchanged in eval mode).
   *
   * @details
   * - **Training mode** (`is_training() == true`): samples a Bernoulli mask
   *   with keep-probability \f$1-p\f$, scales survivors by
   *   \f$\frac{1}{1-p}\f$, and multiplies element-wise with `input`.
   * - **Eval mode** (`is_training() == false`) or when `p == 0`: returns
   *   `input` unchanged (no copy, no mask).
   *
   * @param input  Input tensor of any shape.
   * @return       Masked and scaled tensor in training mode; `input` in eval.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  /**
   * @brief Drop probability \f$p\f$.
   * @return The fraction of elements zeroed per forward call.
   */
  float p() const { return p_; }

 private:
  /// Probability that any single element is zeroed.  Range: [0, 1).
  float p_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_DROPOUT_H_
