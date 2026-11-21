/**
 * @file activations.cc
 * @brief Implementations of neural-network activation modules.
 *
 * @details
 * Each `forward()` method is a thin, one-line dispatch to the corresponding
 * differentiable op in `scorch::ops::activation`.  The ops record the backward
 * function on the autograd graph, so no gradient logic lives here.
 *
 * Op → Activation mapping:
 * | Module     | Op called            |
 * |------------|----------------------|
 * | ReLU       | `scorch::relu()`     |
 * | Sigmoid    | `scorch::sigmoid()`  |
 * | Tanh       | `scorch::tanh_op()`  |
 * | LeakyReLU  | `scorch::leaky_relu()`|
 * | Softmax    | `scorch::softmax()`  |
 */

#include "nn/activations.h"

#include "ops/activation.h"

namespace scorch::nn {

// -----------------------------------------------------------------------------
// ReLU
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> ReLU::forward(const std::shared_ptr<Tensor<float>>& input) {
  // Delegates to the differentiable relu op; backward computes dL/dx = dL/dy *
  // 1[x > 0].
  return scorch::relu(input);
}

// -----------------------------------------------------------------------------
// Sigmoid
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> Sigmoid::forward(const std::shared_ptr<Tensor<float>>& input) {
  // Delegates to the differentiable sigmoid op; backward uses the identity
  // sigma'(x) = sigma(x) * (1 - sigma(x)).
  return scorch::sigmoid(input);
}

// -----------------------------------------------------------------------------
// Tanh
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> Tanh::forward(const std::shared_ptr<Tensor<float>>& input) {
  // Uses tanh_op (not 'tanh' to avoid clash with std::tanh).
  // Backward: 1 - tanh^2(x).
  return scorch::tanh_op(input);
}

// -----------------------------------------------------------------------------
// LeakyReLU
// -----------------------------------------------------------------------------

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

std::shared_ptr<Tensor<float>> LeakyReLU::forward(const std::shared_ptr<Tensor<float>>& input) {
  // Delegates to leaky_relu(input, alpha); backward passes alpha for x < 0,
  // 1 for x >= 0.
  return scorch::leaky_relu(input, negative_slope_);
}

// -----------------------------------------------------------------------------
// Softmax
// -----------------------------------------------------------------------------

Softmax::Softmax(int dim) : dim_(dim) {}

std::shared_ptr<Tensor<float>> Softmax::forward(const std::shared_ptr<Tensor<float>>& input) {
  // scorch::softmax handles the log-sum-exp numerical stability trick
  // internally.
  return scorch::softmax(input, dim_);
}

}  // namespace scorch::nn
