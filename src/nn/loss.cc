/**
 * @file loss.cc
 * @brief Implementation of MSELoss and CrossEntropyLoss modules.
 */

#include "nn/loss.h"

#include <stdexcept>

#include "ops/loss.h"

namespace scorch::nn {

std::shared_ptr<Tensor<float>> MSELoss::forward(const std::shared_ptr<Tensor<float>>& input) {
  throw std::invalid_argument(
      "MSELoss::forward(input) requires two operands: forward(prediction, target)");
}

std::shared_ptr<Tensor<float>> MSELoss::forward(const std::shared_ptr<Tensor<float>>& prediction,
                                                const std::shared_ptr<Tensor<float>>& target) {
  return scorch::mse_loss(prediction, target);
}

std::shared_ptr<Tensor<float>> CrossEntropyLoss::forward(
    const std::shared_ptr<Tensor<float>>& input) {
  throw std::invalid_argument(
      "CrossEntropyLoss::forward(input) requires two operands: forward(logits, targets)");
}

std::shared_ptr<Tensor<float>> CrossEntropyLoss::forward(
    const std::shared_ptr<Tensor<float>>& logits, const std::shared_ptr<Tensor<int>>& targets) {
  return scorch::cross_entropy_loss(logits, targets);
}

}  // namespace scorch::nn
