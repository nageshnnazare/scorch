#ifndef _SCORCH_NN_ACTIVATIONS_H_
#define _SCORCH_NN_ACTIVATIONS_H_

/**
 * @file activations.h
 * @brief Standard neural network activation modules (ReLU, Sigmoid, Tanh, LeakyReLU, Softmax).
 */

#include <memory>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Applies the rectified linear unit function element-wise.
 */
class ReLU : public Module {
 public:
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;
};

/**
 * @brief Applies the Sigmoid function element-wise: 1 / (1 + exp(-x)).
 */
class Sigmoid : public Module {
 public:
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;
};

/**
 * @brief Applies the Hyperbolic Tangent function element-wise: tanh(x).
 */
class Tanh : public Module {
 public:
  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;
};

/**
 * @brief Applies the LeakyReLU function: max(0, x) + negative_slope * min(0, x).
 */
class LeakyReLU : public Module {
 public:
  explicit LeakyReLU(float negative_slope = 0.01f);

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

 private:
  float negative_slope_;
};

/**
 * @brief Applies the Softmax function along specified dimension.
 */
class Softmax : public Module {
 public:
  explicit Softmax(int dim = -1);

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

 private:
  int dim_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_ACTIVATIONS_H_
