#ifndef _SCORCH_OPS_LOSS_H_
#define _SCORCH_OPS_LOSS_H_

/**
 * @file loss.h
 * @brief Differentiable loss functions: MSE Loss, Cross-Entropy Loss.
 */

#include <memory>

#include "tensor.h"

namespace scorch {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& pred,
                                    const std::shared_ptr<Tensor<T>>& target);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& logits,
                                              const std::shared_ptr<Tensor<int>>& targets);

}  // namespace scorch

#endif  // !_SCORCH_OPS_LOSS_H_
