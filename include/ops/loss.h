#ifndef _SCORCH_OPS_LOSS_H_
#define _SCORCH_OPS_LOSS_H_

/**
 * @file loss.h
 * @brief Differentiable loss functions: MSE Loss and Cross-Entropy Loss.
 *
 * @details
 * Provides foundational loss functions for training regression and classification
 * neural network models:
 * - `mse_loss`: Mean Squared Error loss.
 * - `cross_entropy_loss`: Numerically stabilized fused Softmax + Negative Log-Likelihood loss.
 *
 * @ingroup ops
 */

#include <memory>

#include "tensor.h"

namespace scorch {

/**
 * @brief Computes Mean Squared Error (MSE) loss between predictions and targets.
 *
 * @details
 * \f[
 *   \mathcal{L} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
 * \f]
 * Backward gradient:
 * \f[
 *   \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)
 * \f]
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param pred Predicted output tensor of arbitrary shape.
 * @param target Ground-truth target tensor matching shape of `pred`.
 * @return Scalar (0-D or 1-element) loss tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& pred,
                                    const std::shared_ptr<Tensor<T>>& target);

/**
 * @brief Computes categorical Cross-Entropy loss from unnormalized logits and class index targets.
 *
 * @details
 * Fuses Log-Softmax and NLL using the Log-Sum-Exp trick for numerical stability:
 * \f[
 *   \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left( z_{i, c_i} - \log \sum_j e^{z_{i, j}} \right)
 * \f]
 * Backward gradient with respect to logit \f$z_{i, j}\f$:
 * \f[
 *   \frac{\partial \mathcal{L}}{\partial z_{i, j}} = \frac{1}{N} (\text{softmax}(z_i)_j -
 * \mathbf{1}[j = c_i])
 * \f]
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param logits 2D tensor of raw class scores with shape \f$\{N, C\}\f$.
 * @param targets 1D integer tensor containing true class labels \f$\{c_i \in [0, C-1]\}\f$ of shape
 * \f$\{N\}\f$.
 * @return Scalar loss tensor.
 * @throws std::invalid_argument If shapes or dimensions are incompatible.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& logits,
                                              const std::shared_ptr<Tensor<int>>& targets);

}  // namespace scorch

#endif  // !_SCORCH_OPS_LOSS_H_
