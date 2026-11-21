#ifndef _SCORCH_OPTIM_ADAM_H_
#define _SCORCH_OPTIM_ADAM_H_

/**
 * @file adam.h
 * @brief Adam optimizer with adaptive moment estimation.
 *
 * @details
 * Implements the Adam optimization algorithm from Kingma and Ba (2014),
 * matching `torch.optim.Adam`.
 *
 * Update rule:
 * \f[
 *   g_t \leftarrow \nabla_\theta \mathcal{L}(\theta_t) + \lambda \theta_t
 * \f]
 * \f[
 *   m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t
 * \f]
 * \f[
 *   v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
 * \f]
 * \f[
 *   \hat{m}_t \leftarrow \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t \leftarrow \frac{v_t}{1 -
 * \beta_2^t}
 * \f]
 * \f[
 *   \theta_{t+1} \leftarrow \theta_t - \frac{\gamma \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
 * \f]
 *
 * @ingroup optim
 */

#include <cstddef>
#include <memory>
#include <vector>

#include "optim/optimizer.h"

namespace scorch::optim {

/**
 * @brief Adaptive Moment Estimation (Adam) optimizer.
 *
 * Maintains first and second moving average moment estimates for each parameter,
 * adapting learning rates on a per-parameter element basis with bias correction.
 */
class Adam : public Optimizer {
 public:
  /**
   * @brief Construct an Adam optimizer.
   * @param parameters    Collection of parameter tensors to optimize.
   * @param lr            Learning rate (\f$\gamma\f$), default 1e-3.
   * @param beta1         Exponential decay rate for first moment estimates (\f$\beta_1\f$), default
   * 0.9.
   * @param beta2         Exponential decay rate for second moment estimates (\f$\beta_2\f$),
   * default 0.999.
   * @param eps           Term added to denominator to improve numerical stability (\f$\epsilon\f$),
   * default 1e-8.
   * @param weight_decay  L2 regularization penalty factor (\f$\lambda\f$), default 0.0.
   */
  Adam(std::vector<std::shared_ptr<Tensor<float>>> parameters, float lr = 0.001f,
       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

  /**
   * @brief Performs a single Adam parameter update step with bias correction.
   */
  void step() override;

  /**
   * @brief Get the configured learning rate.
   * @return Learning rate float.
   */
  float lr() const { return lr_; }

  /**
   * @brief Get the current step count (number of times step() has been called).
   * @return Current step counter.
   */
  std::size_t step_count() const { return step_count_; }

 private:
  /// Learning rate.
  float lr_;
  /// Beta1 coefficient for first moment moving average.
  float beta1_;
  /// Beta2 coefficient for second moment moving average.
  float beta2_;
  /// Small constant epsilon for numerical stability.
  float eps_;
  /// Weight decay (L2 penalty) factor.
  float weight_decay_;
  /// Running count of completed optimization steps for bias correction.
  std::size_t step_count_{0};
  /// First moment vector per parameter.
  std::vector<std::vector<float>> m_;
  /// Second moment vector per parameter.
  std::vector<std::vector<float>> v_;
};

}  // namespace scorch::optim

#endif  // !_SCORCH_OPTIM_ADAM_H_
