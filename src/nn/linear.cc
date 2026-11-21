/**
 * @file linear.cc
 * @brief Implementation of the fully-connected `Linear` layer.
 *
 * @details
 * ### Initialisation
 * Weights and bias are initialised with **Kaiming uniform** initialisation
 * (also called "He uniform"):
 *
 * \f[
 *   \mathbf{W},\, \mathbf{b} \sim \mathcal{U}\!\left(
 *     -\tfrac{1}{\sqrt{H_{in}}},\; \tfrac{1}{\sqrt{H_{in}}}
 *   \right)
 * \f]
 *
 * This keeps the variance of activations approximately constant across layers
 * when inputs have unit variance, reducing vanishing/exploding gradients at
 * initialisation.
 *
 * ### Forward pass
 * ```
 * [optional unsqueeze if 1-D input]
 *   out = matmul(x, W)        // {N, H_out}
 *   out = out + b              // broadcast bias across rows
 * [optional squeeze if 1-D input]
 * ```
 *
 * All ops are differentiable; the autograd graph is built automatically.
 */

#include "nn/linear.h"

#include <cmath>
#include <stdexcept>

#include "ops/arithmetic.h"
#include "ops/matrix.h"
#include "ops/shape.h"

namespace scorch::nn {

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
  // Kaiming uniform bound: 1 / sqrt(H_in)
  float bound = 1.0f / std::sqrt(static_cast<float>(in_features));

  // Weight matrix W  shape: {H_in, H_out} — stored column-major so that
  // y = x @ W produces {N, H_out} directly.
  weight = register_parameter("weight", scorch::rand({in_features, out_features}, -bound, bound,
                                                     /*requires_grad=*/true));

  if (has_bias_) {
    // Bias vector b  shape: {1, H_out} — broadcast across the batch dimension.
    this->bias = register_parameter("bias", scorch::rand({1, out_features}, -bound, bound,
                                                         /*requires_grad=*/true));
  }
}

// -----------------------------------------------------------------------------
// Forward pass
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> Linear::forward(const std::shared_ptr<Tensor<float>>& input) {
  if (input->shape.empty()) {
    throw std::invalid_argument("Linear::forward: input tensor has an empty shape");
  }

  // ── Handle 1-D input ──────────────────────────────────────────────────────
  // A bare 1-D vector of shape {H_in} is unsqueezed to {1, H_in} so that the
  // matrix-multiply is well-defined.  The extra batch dimension is squeezed
  // back off before returning.
  bool unsqueezed = false;
  std::shared_ptr<Tensor<float>> x = input;

  if (x->shape.size() == 1) {
    if (x->shape[0] != in_features_) {
      throw std::invalid_argument("Linear::forward: 1-D input size does not match in_features");
    }
    x = unsqueeze(x, 0);  // {H_in} → {1, H_in}
    unsqueezed = true;

  } else if (x->shape.size() == 2) {
    if (x->shape[1] != in_features_) {
      throw std::invalid_argument(
          "Linear::forward: input feature dimension does not match in_features");
    }

  } else {
    throw std::invalid_argument(
        "Linear::forward: only 1-D {H_in} and 2-D {N, H_in} inputs are "
        "supported");
  }

  // ── Affine transformation ─────────────────────────────────────────────────
  // y = x @ W  →  shape {N, H_out}
  auto out = matmul(x, weight);

  // y = y + b  (bias is {1, H_out}, broadcast across N rows)
  if (has_bias_ && bias) {
    out = out + bias;
  }

  // ── Undo unsqueeze for 1-D inputs ─────────────────────────────────────────
  if (unsqueezed) {
    out = squeeze(out, 0);  // {1, H_out} → {H_out}
  }

  return out;
}

}  // namespace scorch::nn
