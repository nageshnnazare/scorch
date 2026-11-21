#ifndef _SCORCH_OPTIM_SGD_H_
#define _SCORCH_OPTIM_SGD_H_

/**
 * @file sgd.h
 * @brief Stochastic Gradient Descent optimizer with momentum and weight decay.
 *
 * @details
 * Implements SGD with optional classical momentum and L2 penalty (weight decay),
 * matching `torch.optim.SGD`.
 *
 * Update rule:
 * \f[
 *   g_t \leftarrow \nabla_\theta \mathcal{L}(\theta_t) + \lambda \theta_t
 * \f]
 * \f[
 *   v_t \leftarrow \mu v_{t-1} + g_t \quad (\text{if } \mu > 0, \text{ else } g_t)
 * \f]
 * \f[
 *   \theta_{t+1} \leftarrow \theta_t - \gamma v_t
 * \f]
 * where \f$\gamma\f$ is learning rate, \f$\mu\f$ is momentum, and \f$\lambda\f$ is weight decay.
 *
 * @ingroup optim
 */

#include <memory>
#include <vector>

#include "optim/optimizer.h"

namespace scorch::optim {

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 *
 * Supports momentum buffer tracking and L2 weight regularization.
 */
class SGD : public Optimizer {
 public:
  /**
   * @brief Construct an SGD optimizer.
   * @param parameters    Collection of parameter tensors to optimize.
   * @param lr            Learning rate (\f$\gamma\f$). Must be > 0.
   * @param momentum      Momentum factor (\f$\mu\f$), default 0.0 (no momentum).
   * @param weight_decay  L2 regularization penalty factor (\f$\lambda\f$), default 0.0.
   */
  SGD(std::vector<std::shared_ptr<Tensor<float>>> parameters, float lr, float momentum = 0.0f,
      float weight_decay = 0.0f);

  /**
   * @brief Performs a single SGD parameter update step.
   *
   * Iterates through parameters, computes momentum updates if enabled, and
   * updates parameter values in-place.
   */
  void step() override;

  /**
   * @brief Get the configured learning rate.
   * @return Learning rate float.
   */
  float lr() const { return lr_; }

  /**
   * @brief Get the configured momentum factor.
   * @return Momentum float.
   */
  float momentum() const { return momentum_; }

  /**
   * @brief Get the configured weight decay factor.
   * @return Weight decay float.
   */
  float weight_decay() const { return weight_decay_; }

 private:
  /// Learning rate.
  float lr_;
  /// Momentum factor.
  float momentum_;
  /// Weight decay (L2 penalty) factor.
  float weight_decay_;
  /// Velocity buffer per parameter for momentum tracking.
  std::vector<std::vector<float>> velocities_;
};

}  // namespace scorch::optim

#endif  // !_SCORCH_OPTIM_SGD_H_
