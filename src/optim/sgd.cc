/**
 * @file sgd.cc
 * @brief Implementation of Stochastic Gradient Descent optimizer.
 *
 * @details
 * Performs in-place parameter updates with optional momentum buffer tracking
 * and L2 weight decay regularisation:
 * \f[
 *   g_t \leftarrow g_t + \lambda \theta_t
 * \f]
 * \f[
 *   v_t \leftarrow \mu v_{t-1} + g_t
 * \f]
 * \f[
 *   \theta_t \leftarrow \theta_{t-1} - \gamma v_t
 * \f]
 */

#include "optim/sgd.h"

namespace scorch::optim {

SGD::SGD(std::vector<std::shared_ptr<Tensor<float>>> parameters, float lr, float momentum,
         float weight_decay)
    : Optimizer(std::move(parameters)), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {
  velocities_.resize(parameters_.size());
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    if (parameters_[i]) {
      velocities_[i].assign(parameters_[i]->data.size(), 0.0f);
    }
  }
}

void SGD::step() {
  for (std::size_t p_idx = 0; p_idx < parameters_.size(); ++p_idx) {
    auto& param = parameters_[p_idx];
    if (!param || param->grad.empty()) continue;

    auto& v = velocities_[p_idx];
    if (v.size() != param->data.size()) {
      v.assign(param->data.size(), 0.0f);
    }

    for (std::size_t i = 0; i < param->data.size(); ++i) {
      float d_p = param->grad[i];

      if (weight_decay_ != 0.0f) {
        d_p += weight_decay_ * param->data[i];
      }

      if (momentum_ != 0.0f) {
        v[i] = momentum_ * v[i] + d_p;
        d_p = v[i];
      }

      param->data[i] -= lr_ * d_p;
    }
  }
}

}  // namespace scorch::optim
