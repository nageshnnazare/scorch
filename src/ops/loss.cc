/**
 * @file loss.cc
 * @brief Implementation of loss functions: MSE Loss, Cross-Entropy Loss.
 */

#include "ops/loss.h"

#include <cmath>
#include <stdexcept>

#include "ops/ops_common.h"

namespace AG {

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

template <TensorScalar T>
std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& logits,
                                              const std::shared_ptr<Tensor<int>>& targets) {
  if (logits->shape.size() != 2) {
    throw std::invalid_argument("cross_entropy_loss: logits must be rank-2 {N, C}");
  }
  if (targets->shape.size() != 1 || targets->shape[0] != logits->shape[0]) {
    throw std::invalid_argument("cross_entropy_loss: targets must have shape {N}");
  }

  const std::size_t batch = logits->shape[0];
  const std::size_t classes = logits->shape[1];
  bool rg = logits->requires_grad;
  auto out = detail::make_result<T>({1}, rg, "cross_entropy_loss", {logits});

  double total_loss = 0.0;
  for (std::size_t i = 0; i < batch; ++i) {
    int ti = targets->data[i];
    double max_val = static_cast<double>(logits->data[i * classes]);
    for (std::size_t j = 1; j < classes; ++j) {
      double v = static_cast<double>(logits->data[i * classes + j]);
      if (v > max_val) max_val = v;
    }
    double log_sum_exp = 0.0;
    for (std::size_t j = 0; j < classes; ++j) {
      log_sum_exp += std::exp(static_cast<double>(logits->data[i * classes + j]) - max_val);
    }
    log_sum_exp = max_val + std::log(log_sum_exp);
    total_loss += -static_cast<double>(logits->data[i * classes + ti]) + log_sum_exp;
  }
  out->data[0] = static_cast<T>(total_loss / static_cast<double>(batch));

  if (rg) {
    auto target_data = targets->data;
    out->backward_fn = [logits, target_data, batch, classes]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*logits);

      if (logits->requires_grad) {
        float g = self->grad[0];
        float inv_n = 1.f / static_cast<float>(batch);

        for (std::size_t i = 0; i < batch; ++i) {
          double max_val = static_cast<double>(logits->data[i * classes]);
          for (std::size_t j = 1; j < classes; ++j) {
            double v = static_cast<double>(logits->data[i * classes + j]);
            if (v > max_val) max_val = v;
          }
          double sum_exp = 0.0;
          std::vector<float> sm(classes);
          for (std::size_t j = 0; j < classes; ++j) {
            sm[j] = static_cast<float>(
                std::exp(static_cast<double>(logits->data[i * classes + j]) - max_val));
            sum_exp += sm[j];
          }
          for (std::size_t j = 0; j < classes; ++j) {
            sm[j] /= static_cast<float>(sum_exp);
          }

          int ti = target_data[i];
          for (std::size_t j = 0; j < classes; ++j) {
            float indicator = (static_cast<int>(j) == ti) ? 1.f : 0.f;
            logits->grad[i * classes + j] += g * (sm[j] - indicator) * inv_n;
          }
        }
      }
    };
  }

  return out;
}

// Explicit Instantiations
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

}  // namespace AG
