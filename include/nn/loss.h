#ifndef _SCORCH_NN_LOSS_H_
#define _SCORCH_NN_LOSS_H_

/**
 * @file loss.h
 * @brief Loss function modules (MSELoss, CrossEntropyLoss).
 */

#include <memory>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Mean Squared Error loss module.
 */
class MSELoss : public Module {
 public:
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& prediction,
                                         const std::shared_ptr<Tensor<float>>& target);
};

/**
 * @brief Cross-Entropy loss module.
 */
class CrossEntropyLoss : public Module {
 public:
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& logits,
                                         const std::shared_ptr<Tensor<int>>& targets);
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_LOSS_H_
