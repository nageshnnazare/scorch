/**
 * @file dropout.cc
 * @brief Implementation of Dropout module.
 */

#include "nn/dropout.h"

#include <random>
#include <stdexcept>

#include "ops/arithmetic.h"

namespace scorch::nn {

Dropout::Dropout(float p) : p_(p) {
  if (p_ < 0.0f || p_ >= 1.0f) {
    throw std::invalid_argument("Dropout: probability p must be in [0, 1)");
  }
}

std::shared_ptr<Tensor<float>> Dropout::forward(const std::shared_ptr<Tensor<float>>& input) {
  if (!is_training() || p_ <= 0.0f) {
    return input;
  }

  // Inverted dropout: during training, scale kept activations by 1 / (1 - p)
  float scale = 1.0f / (1.0f - p_);
  std::vector<float> mask_data(input->data.size());

  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::bernoulli_distribution dist(1.0f - p_);

  for (std::size_t i = 0; i < mask_data.size(); ++i) {
    mask_data[i] = dist(gen) ? scale : 0.0f;
  }

  auto mask = scorch::from(input->shape, std::move(mask_data), /*requires_grad=*/false);
  return input * mask;
}

}  // namespace scorch::nn
