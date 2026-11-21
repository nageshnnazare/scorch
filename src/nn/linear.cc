/**
 * @file linear.cc
 * @brief Implementation of Linear fully-connected layer.
 */

#include "nn/linear.h"

#include <cmath>
#include <stdexcept>

#include "ops/arithmetic.h"
#include "ops/matrix.h"
#include "ops/shape.h"

namespace scorch::nn {

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
  float bound = 1.0f / std::sqrt(static_cast<float>(in_features));

  // Initialize weights: shape {in_features, out_features}
  weight = register_parameter(
      "weight", scorch::rand({in_features, out_features}, -bound, bound, /*requires_grad=*/true));

  if (has_bias_) {
    // Initialize bias: shape {1, out_features}
    this->bias = register_parameter(
        "bias", scorch::rand({1, out_features}, -bound, bound, /*requires_grad=*/true));
  }
}

std::shared_ptr<Tensor<float>> Linear::forward(const std::shared_ptr<Tensor<float>>& input) {
  if (input->shape.empty()) {
    throw std::invalid_argument("Linear::forward: input tensor has empty shape");
  }

  // Support 1D input by unsqueezing to {1, in_features}
  bool unsqueezed = false;
  std::shared_ptr<Tensor<float>> x = input;
  if (x->shape.size() == 1) {
    if (x->shape[0] != in_features_) {
      throw std::invalid_argument("Linear::forward: input feature size mismatch");
    }
    x = unsqueeze(x, 0);
    unsqueezed = true;
  } else if (x->shape.size() == 2) {
    if (x->shape[1] != in_features_) {
      throw std::invalid_argument("Linear::forward: input feature size mismatch");
    }
  } else {
    throw std::invalid_argument("Linear::forward: only 1D and 2D inputs are supported");
  }

  // y = x @ W
  auto out = matmul(x, weight);

  // y = y + b (broadcasted across batch rows)
  if (has_bias_ && bias) {
    out = out + bias;
  }

  if (unsqueezed) {
    out = squeeze(out, 0);
  }

  return out;
}

}  // namespace scorch::nn
