#ifndef _SCORCH_NN_ACTIVATIONS_H_
#define _SCORCH_NN_ACTIVATIONS_H_

/**
 * @file activations.h
 * @brief Activation-function modules for neural networks.
 *
 * @details
 * Each class wraps a differentiable element-wise (or dimension-wise) operation
 * as a `scorch::nn::Module`, making it composable inside `Sequential` or any
 * custom network.  All operations are autograd-aware; calling `forward()`
 * records the necessary gradient functions automatically.
 *
 * | Class      | Formula                                                      | PyTorch equiv |
 * |------------|--------------------------------------------------------------|-------------------------|
 * | ReLU       | \f$\max(0,x)\f$                                              | `nn.ReLU()` | |
 * Sigmoid    | \f$\frac{1}{1+e^{-x}}\f$                                     | `nn.Sigmoid()` | |
 * Tanh       | \f$\tanh(x)\f$                                               | `nn.Tanh()` | |
 * LeakyReLU  | \f$\max(\alpha x,\ x),\ \alpha=0.01\f$ (default)             | `nn.LeakyReLU()` | |
 * Softmax    | \f$\frac{e^{x_i}}{\sum_j e^{x_j}}\f$ along `dim`             | `nn.Softmax(dim=…)` |
 *
 * @ingroup nn
 */

#include <memory>

#include "nn/module.h"

namespace scorch::nn {

// =============================================================================
// ReLU
// =============================================================================

/**
 * @brief Rectified Linear Unit activation applied element-wise.
 *
 * @details
 * Computes:
 * \f[
 *   \text{ReLU}(x) = \max(0,\, x)
 * \f]
 *
 * ReLU is the most widely used activation in deep networks.  It is
 * computationally cheap, mitigates the vanishing-gradient problem for positive
 * activations, and induces sparsity in hidden representations.
 *
 * **Gradient:** \f$\partial y / \partial x = \mathbf{1}[x > 0]\f$.
 *
 * ### Usage
 * @code{.cpp}
 * auto relu = std::make_shared<scorch::nn::ReLU>();
 * auto y = relu->forward(x);    // or: (*relu)(x)
 * @endcode
 */
class ReLU : public Module {
 public:
  /**
   * @brief Apply ReLU element-wise.
   * @param input  Input tensor of any shape.
   * @return       Tensor of the same shape; negative values clamped to 0.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;
};

// =============================================================================
// Sigmoid
// =============================================================================

/**
 * @brief Sigmoid (logistic) activation applied element-wise.
 *
 * @details
 * Computes:
 * \f[
 *   \sigma(x) = \frac{1}{1 + e^{-x}}
 * \f]
 *
 * Maps all real values into \f$(0,\, 1)\f$, making it suitable for binary
 * classification output layers and gating mechanisms.
 *
 * **Gradient:** \f$\sigma'(x) = \sigma(x)(1 - \sigma(x))\f$.
 *
 * @note Sigmoid can suffer from vanishing gradients for very large or very
 * small inputs.  Prefer ReLU or LeakyReLU for hidden layers.
 *
 * ### Usage
 * @code{.cpp}
 * auto sig = std::make_shared<scorch::nn::Sigmoid>();
 * auto prob = sig->forward(logit);   // outputs in (0, 1)
 * @endcode
 */
class Sigmoid : public Module {
 public:
  /**
   * @brief Apply Sigmoid element-wise.
   * @param input  Input tensor of any shape.
   * @return       Tensor of the same shape; values in \f$(0, 1)\f$.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;
};

// =============================================================================
// Tanh
// =============================================================================

/**
 * @brief Hyperbolic Tangent activation applied element-wise.
 *
 * @details
 * Computes:
 * \f[
 *   \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
 * \f]
 *
 * Maps all real values into \f$(-1,\, 1)\f$.  Zero-centred output makes it
 * preferable to Sigmoid for hidden layers in many architectures (e.g. RNNs).
 *
 * **Gradient:** \f$\tanh'(x) = 1 - \tanh^2(x)\f$.
 *
 * ### Usage
 * @code{.cpp}
 * auto tanh_act = std::make_shared<scorch::nn::Tanh>();
 * auto h = tanh_act->forward(z);
 * @endcode
 */
class Tanh : public Module {
 public:
  /**
   * @brief Apply tanh element-wise.
   * @param input  Input tensor of any shape.
   * @return       Tensor of the same shape; values in \f$(-1, 1)\f$.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;
};

// =============================================================================
// LeakyReLU
// =============================================================================

/**
 * @brief Leaky Rectified Linear Unit: allows a small, non-zero gradient for
 *        negative inputs.
 *
 * @details
 * Computes:
 * \f[
 *   \text{LeakyReLU}(x) =
 *   \begin{cases}
 *     x                   & \text{if } x \geq 0 \\
 *     \alpha \cdot x      & \text{if } x < 0
 *   \end{cases}
 * \f]
 *
 * where \f$\alpha\f$ (`negative_slope`) defaults to `0.01`.
 *
 * Unlike ReLU, LeakyReLU never completely kills a neuron; neurons with large
 * negative pre-activations still receive a small gradient
 * \f$(\alpha)\f$ and can recover during training.
 *
 * **Gradient:** \f$\partial y / \partial x = \mathbf{1}[x \geq 0] + \alpha \cdot \mathbf{1}[x <
 * 0]\f$.
 *
 * ### Usage
 * @code{.cpp}
 * // Default slope 0.01
 * auto lrelu = std::make_shared<scorch::nn::LeakyReLU>();
 * // Custom slope
 * auto lrelu2 = std::make_shared<scorch::nn::LeakyReLU>(0.2f);
 * @endcode
 */
class LeakyReLU : public Module {
 public:
  /**
   * @brief Construct a LeakyReLU module.
   * @param negative_slope  Slope of the function for negative inputs
   *                        \f$\alpha\f$.  Must be non-negative.
   *                        Default: `0.01`.
   */
  explicit LeakyReLU(float negative_slope = 0.01f);

  /**
   * @brief Apply LeakyReLU element-wise.
   * @param input  Input tensor of any shape.
   * @return       Tensor of the same shape.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

 private:
  /// Slope applied to negative pre-activations \f$\alpha\f$.
  float negative_slope_;
};

// =============================================================================
// Softmax
// =============================================================================

/**
 * @brief Softmax normalisation along a specified dimension.
 *
 * @details
 * Computes:
 * \f[
 *   \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
 * \f]
 *
 * applied along dimension `dim`.  The output sums to 1 along that dimension,
 * yielding a valid probability distribution.
 *
 * **Numerical stability:** the implementation subtracts \f$\max(x)\f$ before
 * exponentiation to prevent overflow (log-sum-exp trick).
 *
 * @note For classification tasks, prefer `CrossEntropyLoss` (which fuses
 *       log-softmax + NLL for numerical stability) rather than applying
 *       Softmax manually before the loss.
 *
 * ### Usage
 * @code{.cpp}
 * // Softmax over class dimension (dim=1 for shape {N, C})
 * auto sm = std::make_shared<scorch::nn::Softmax>(1);
 * auto probs = sm->forward(logits);  // sums to 1 along dim 1
 * @endcode
 */
class Softmax : public Module {
 public:
  /**
   * @brief Construct a Softmax module.
   * @param dim  Dimension along which Softmax is computed.
   *             `-1` (default) uses the last dimension.
   */
  explicit Softmax(int dim = -1);

  /**
   * @brief Apply Softmax along the configured dimension.
   * @param input  Input tensor of any shape.
   * @return       Tensor of the same shape; sums to 1 along `dim`.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

 private:
  /// The axis along which the softmax normalization is applied.
  int dim_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_ACTIVATIONS_H_
