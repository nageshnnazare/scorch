/**
 * @file adam.cc
 * @brief Implementation of Adam optimizer.
 */

#include "optim/adam.h"

#include <cmath>

namespace scorch::optim {

Adam::Adam(std::vector<std::shared_ptr<Tensor<float>>> parameters, float lr, float beta1,
           float beta2, float eps, float weight_decay)
    : Optimizer(std::move(parameters)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {
  m_.resize(parameters_.size());
  v_.resize(parameters_.size());
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    if (parameters_[i]) {
      m_[i].assign(parameters_[i]->data.size(), 0.0f);
      v_[i].assign(parameters_[i]->data.size(), 0.0f);
    }
  }
}

void Adam::step() {
  step_count_++;
  float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
  float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

  for (std::size_t p_idx = 0; p_idx < parameters_.size(); ++p_idx) {
    auto& param = parameters_[p_idx];
    if (!param || param->grad.empty()) continue;

    auto& m = m_[p_idx];
    auto& v = v_[p_idx];
    if (m.size() != param->data.size()) m.assign(param->data.size(), 0.0f);
    if (v.size() != param->data.size()) v.assign(param->data.size(), 0.0f);

    for (std::size_t i = 0; i < param->data.size(); ++i) {
      float grad = param->grad[i];

      if (weight_decay_ != 0.0f) {
        grad += weight_decay_ * param->data[i];
      }

      // Exponential moving average of gradient
      m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad;

      // Exponential moving average of squared gradient
      v[i] = beta2_ * v[i] + (1.0f - beta2_) * grad * grad;

      float m_hat = m[i] / bias_correction1;
      float v_hat = v[i] / bias_correction2;

      param->data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

}  // namespace scorch::optim
