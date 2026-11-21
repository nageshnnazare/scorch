/**
 * @file loss.cc
 * @brief Implementation of loss functions: MSE Loss, Cross-Entropy Loss.
 *
 * @details
 * Defines forward computations and autograd backward closures for:
 * - `mse_loss`: Mean Squared Error loss between prediction and target tensors.
 * - `cross_entropy_loss`: Fused Log-Softmax + NLL classification loss using the
 *   log-sum-exp trick for numerical stability.
 */

#include "ops/loss.h"

#include <cmath>
#include <stdexcept>

#include "ops/ops_common.h"

namespace scorch {

/**
 * @brief Computes Mean Squared Error (MSE) loss with autograd gradients.
 * @param pred Predicted tensor.
 * @param target Ground-truth tensor with matching shape.
 * @return 1-element tensor holding scalar MSE loss.
 * @throws std::invalid_argument If shapes of pred and target mismatch.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& pred,
                                    const std::shared_ptr<Tensor<T>>& target) {
  if (pred->shape != target->shape) {
    throw std::invalid_argument("mse_loss: shape mismatch");
  }

  bool rg = pred->requires_grad || target->requires_grad;
  std::size_t n = pred->data.size();
  auto out = detail::make_result<T>({1}, rg, "mse_loss", {pred, target});

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(pred->data[i]) - static_cast<double>(target->data[i]);
    sum += diff * diff;
  }
  out->data[0] = static_cast<T>(sum / static_cast<double>(n));

  if (rg) {
    out->backward_fn = [pred, target, n]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*pred);
      detail::ensure_grad(*target);

      float g = self->grad[0];
      float inv_n = 1.f / static_cast<float>(n);

      for (std::size_t i = 0; i < n; ++i) {
        float diff = static_cast<float>(pred->data[i]) - static_cast<float>(target->data[i]);
        if (pred->requires_grad) {
          pred->grad[i] += g * 2.f * diff * inv_n;
        }
        if (target->requires_grad) {
          target->grad[i] += g * -2.f * diff * inv_n;
        }
      }
    };
  }

  return out;
}

/**
 * @brief Computes fused Softmax + NLL Categorical Cross-Entropy loss.
 * @param logits 2D unnormalized scores tensor of shape {N, C}.
 * @param targets 1D integer class labels of shape {N}.
 * @return 1-element scalar loss tensor.
 * @throws std::invalid_argument If logits is not 2D or targets shape != {N}.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& logits,
                                              const std::shared_ptr<Tensor<int>>& targets) {
  if (logits->shape.size() != 2) {
    throw std::invalid_argument("cross_entropy_loss: logits must be 2D {N, C}");
  }
  std::size_t n = logits->shape[0];
  std::size_t c = logits->shape[1];
  if (targets->shape.size() != 1 || targets->shape[0] != n) {
    throw std::invalid_argument("cross_entropy_loss: targets must be 1D {N}");
  }

  bool rg = logits->requires_grad;
  auto out = detail::make_result<T>({1}, rg, "cross_entropy_loss", {logits});

  std::vector<double> probs(n * c);
  double total_loss = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    double max_val = static_cast<double>(logits->data[i * c]);
    for (std::size_t j = 1; j < c; ++j) {
      double v = static_cast<double>(logits->data[i * c + j]);
      if (v > max_val) max_val = v;
    }

    double sum_exp = 0.0;
    for (std::size_t j = 0; j < c; ++j) {
      double p = std::exp(static_cast<double>(logits->data[i * c + j]) - max_val);
      probs[i * c + j] = p;
      sum_exp += p;
    }

    for (std::size_t j = 0; j < c; ++j) {
      probs[i * c + j] /= sum_exp;
    }

    int target_class = targets->data[i];
    if (target_class < 0 || static_cast<std::size_t>(target_class) >= c) {
      throw std::invalid_argument("cross_entropy_loss: target label out of range");
    }
    double log_prob = std::log(probs[i * c + target_class]);
    total_loss -= log_prob;
  }

  out->data[0] = static_cast<T>(total_loss / static_cast<double>(n));

  if (rg) {
    out->backward_fn = [logits, targets, probs = std::move(probs), n, c]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*logits);

      if (logits->requires_grad) {
        float g = self->grad[0];
        float inv_n = 1.f / static_cast<float>(n);

        for (std::size_t i = 0; i < n; ++i) {
          int target_class = targets->data[i];
          for (std::size_t j = 0; j < c; ++j) {
            float p = static_cast<float>(probs[i * c + j]);
            float grad = (j == static_cast<std::size_t>(target_class)) ? (p - 1.f) : p;
            logits->grad[i * c + j] += g * grad * inv_n;
          }
        }
      }
    };
  }

  return out;
}

// Explicit template instantiations
template std::shared_ptr<Tensor<float>> mse_loss(const std::shared_ptr<Tensor<float>>&,
                                                 const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> mse_loss(const std::shared_ptr<Tensor<double>>&,
                                                  const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> mse_loss(const std::shared_ptr<Tensor<int>>&,
                                               const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> cross_entropy_loss(const std::shared_ptr<Tensor<float>>&,
                                                           const std::shared_ptr<Tensor<int>>&);
template std::shared_ptr<Tensor<double>> cross_entropy_loss(const std::shared_ptr<Tensor<double>>&,
                                                            const std::shared_ptr<Tensor<int>>&);

}  // namespace scorch
