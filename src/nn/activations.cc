/**
 * @file activations.cc
 * @brief Implementation of neural network activation modules.
 */

#include "nn/activations.h"

#include "ops/activation.h"

namespace scorch::nn {

std::shared_ptr<Tensor<float>> ReLU::forward(const std::shared_ptr<Tensor<float>>& input) {
  return scorch::relu(input);
}

std::shared_ptr<Tensor<float>> Sigmoid::forward(const std::shared_ptr<Tensor<float>>& input) {
  return scorch::sigmoid(input);
}

std::shared_ptr<Tensor<float>> Tanh::forward(const std::shared_ptr<Tensor<float>>& input) {
  return scorch::tanh_op(input);
}

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

std::shared_ptr<Tensor<float>> LeakyReLU::forward(const std::shared_ptr<Tensor<float>>& input) {
  return scorch::leaky_relu(input, negative_slope_);
}

Softmax::Softmax(int dim) : dim_(dim) {}

std::shared_ptr<Tensor<float>> Softmax::forward(const std::shared_ptr<Tensor<float>>& input) {
  return scorch::softmax(input, dim_);
}

}  // namespace scorch::nn
