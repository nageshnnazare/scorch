#ifndef _SCORCH_NN_LINEAR_H_
#define _SCORCH_NN_LINEAR_H_

/**
 * @file linear.h
 * @brief Fully-connected (dense) linear transformation layer.
 *
 * @details
 * `scorch::nn::Linear` applies an affine transformation to the incoming data:
 *
 * \f[
 *   \mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}
 * \f]
 *
 * where:
 * - \f$\mathbf{x} \in \mathbb{R}^{N \times H_{in}}\f$ is the input batch,
 * - \f$\mathbf{W} \in \mathbb{R}^{H_{in} \times H_{out}}\f$ is the weight
 *   matrix (`weight`), and
 * - \f$\mathbf{b} \in \mathbb{R}^{1 \times H_{out}}\f$ is the bias vector
 *   (`bias`), broadcast across the batch.
 *
 * This is equivalent to `torch.nn.Linear` in PyTorch.
 *
 * ### Weight initialisation
 * Weights and bias are initialised with **Kaiming uniform** initialisation
 * (uniform in \f$[-\frac{1}{\sqrt{H_{in}}},\ \frac{1}{\sqrt{H_{in}}}]\f$),
 * the same default used by PyTorch.
 *
 * ### Usage
 * @code{.cpp}
 * // 784 → 256 → 10 two-layer network
 * auto fc1 = std::make_shared<scorch::nn::Linear>(784, 256);
 * auto fc2 = std::make_shared<scorch::nn::Linear>(256, 10);
 *
 * auto h = scorch::relu(fc1->forward(x));  // {batch, 256}
 * auto y = fc2->forward(h);                // {batch, 10}
 * @endcode
 *
 * @ingroup nn
 */

#include <cstddef>
#include <memory>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Applies a learnable affine transformation: **y = x W + b**.
 *
 * @details
 * The layer stores two public member tensors:
 * - `weight` — shape `{in_features, out_features}`, always present.
 * - `bias`   — shape `{1, out_features}`, present only when `bias = true`.
 *
 * Both are automatically registered as parameters and included in
 * `Module::parameters()`.
 *
 * **Shapes accepted by `forward()`:**
 * | Input shape              | Output shape               |
 * |--------------------------|----------------------------|
 * | `{in_features}`          | `{out_features}` (squeezed)|
 * | `{N, in_features}`       | `{N, out_features}`        |
 *
 * @note Only 1-D and 2-D inputs are currently supported.  Higher-rank inputs
 *       should be reshaped to `{N, in_features}` before passing in.
 */
class Linear : public Module {
 public:
  /**
   * @brief Construct a Linear layer.
   *
   * @details
   * Allocates and initialises the weight matrix (and optionally the bias
   * vector) using Kaiming uniform initialisation.
   *
   * @param in_features   Number of features in each input sample \f$H_{in}\f$.
   * @param out_features  Number of features in each output sample \f$H_{out}\f$.
   * @param bias          If `true` (default) a learnable bias is added to the
   *                      output.  Set to `false` to omit the bias term.
   */
  Linear(std::size_t in_features, std::size_t out_features, bool bias = true);

  /**
   * @brief Compute **y = x W + b** and return the result.
   *
   * @details
   * Steps performed internally:
   * 1. If `input` is 1-D, unsqueeze it to `{1, in_features}`.
   * 2. Matrix-multiply: `out = matmul(x, weight)` → `{N, out_features}`.
   * 3. Add bias (broadcasted): `out = out + bias` if bias is enabled.
   * 4. If the input was 1-D, squeeze the batch dimension back off.
   *
   * All operations are differentiable; the autograd graph is built
   * automatically.
   *
   * @param input  Input tensor of shape `{in_features}` or `{N, in_features}`.
   * @return       Output tensor of shape `{out_features}` or `{N, out_features}`.
   *
   * @throws std::invalid_argument If the input is empty, has the wrong feature
   *         dimension, or has more than 2 dimensions.
   */
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------

  /// @brief Number of input features \f$H_{in}\f$.
  std::size_t in_features() const { return in_features_; }

  /// @brief Number of output features \f$H_{out}\f$.
  std::size_t out_features() const { return out_features_; }

  /// @brief Whether a bias term is included in this layer.
  bool has_bias() const { return has_bias_; }

  // ---------------------------------------------------------------------------
  // Public parameters (directly accessible like PyTorch)
  // ---------------------------------------------------------------------------

  /**
   * @brief Learnable weight matrix of shape `{in_features, out_features}`.
   *
   * Initialised with Kaiming uniform in
   * \f$\left[-\tfrac{1}{\sqrt{H_{in}}},\ \tfrac{1}{\sqrt{H_{in}}}\right]\f$.
   */
  std::shared_ptr<Tensor<float>> weight;

  /**
   * @brief Learnable bias vector of shape `{1, out_features}`.
   *
   * `nullptr` when the layer was constructed with `bias = false`.
   * Initialised with the same Kaiming uniform bounds as `weight`.
   */
  std::shared_ptr<Tensor<float>> bias;

 private:
  std::size_t in_features_;   ///< Cached input feature count.
  std::size_t out_features_;  ///< Cached output feature count.
  bool has_bias_;             ///< Whether a bias is registered.
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_LINEAR_H_
