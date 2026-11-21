/**
 * @file activation.cc
 * @brief Implementation of activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax.
 *
 * @details
 * Defines the forward evaluation and reverse-mode gradient closures for:
 * - `relu`: y = max(0, x), grad = 1[x > 0]
 * - `leaky_relu`: y = max(alpha * x, x), grad = 1[x > 0] + alpha * 1[x <= 0]
 * - `sigmoid`: y = 1 / (1 + exp(-x)), grad = y * (1 - y)
 * - `tanh_op`: y = tanh(x), grad = 1 - y^2
 * - `softmax`: y = exp(x - max(x)) / sum(exp(x - max(x)))
 */

#include "ops/activation.h"

#include <cmath>
#include <stdexcept>

#include "ops/ops_common.h"

namespace scorch {

/**
 * @brief Differentiable ReLU element-wise operation.
 * @param x Input tensor.
 * @return Tensor with negative elements clamped to 0.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "relu", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = (x->data[i] > T(0)) ? x->data[i] : T(0);
  }

  if (rg) {
    out->backward_fn = [x]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          if (x->data[i] > T(0)) {
            x->grad[i] += self->grad[i];
          }
        }
      }
    };
  }

  return out;
}

/**
 * @brief Differentiable Leaky ReLU element-wise operation.
 * @param x Input tensor.
 * @param alpha Negative slope multiplier.
 * @return Tensor with negative elements scaled by alpha.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> leaky_relu(const std::shared_ptr<Tensor<T>>& x, T alpha) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "leaky_relu", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = (x->data[i] > T(0)) ? x->data[i] : alpha * x->data[i];
  }

  if (rg) {
    out->backward_fn = [x, alpha]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float slope = (x->data[i] > T(0)) ? 1.f : static_cast<float>(alpha);
          x->grad[i] += self->grad[i] * slope;
        }
      }
    };
  }

  return out;
}

/**
 * @brief Differentiable Sigmoid element-wise operation.
 * @param x Input tensor.
 * @return Tensor with elements mapped to (0, 1).
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "sigmoid", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    double v = static_cast<double>(x->data[i]);
    out->data[i] = static_cast<T>(1.0 / (1.0 + std::exp(-v)));
  }

  if (rg) {
    out->backward_fn = [x, out]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float s = static_cast<float>(out->data[i]);
          x->grad[i] += self->grad[i] * s * (1.f - s);
        }
      }
    };
  }

  return out;
}

/**
 * @brief Differentiable Hyperbolic Tangent (Tanh) element-wise operation.
 * @param x Input tensor.
 * @return Tensor with elements mapped to (-1, 1).
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> tanh_op(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "tanh", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    double v = static_cast<double>(x->data[i]);
    out->data[i] = static_cast<T>(std::tanh(v));
  }

  if (rg) {
    out->backward_fn = [x, out]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float th = static_cast<float>(out->data[i]);
          x->grad[i] += self->grad[i] * (1.f - th * th);
        }
      }
    };
  }

  return out;
}

/**
 * @brief Differentiable Softmax along a specified dimension.
 * @param x Input tensor.
 * @param dim Axis along which to compute softmax normalization.
 * @return Normalized tensor summing to 1 along dim.
 * @throws std::invalid_argument If dim >= x->shape.size().
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& x, std::size_t dim) {
  if (dim >= x->shape.size()) {
    throw std::invalid_argument("softmax: dim out of range");
  }

  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "softmax", {x});

  std::size_t outer = 1;
  for (std::size_t i = 0; i < dim; ++i) outer *= x->shape[i];
  std::size_t dim_size = x->shape[dim];
  std::size_t inner = 1;
  for (std::size_t i = dim + 1; i < x->shape.size(); ++i) inner *= x->shape[i];

  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t inn = 0; inn < inner; ++inn) {
      double max_val = static_cast<double>(x->data[o * dim_size * inner + 0 * inner + inn]);
      for (std::size_t d = 1; d < dim_size; ++d) {
        double v = static_cast<double>(x->data[o * dim_size * inner + d * inner + inn]);
        if (v > max_val) max_val = v;
      }

      double sum_exp = 0.0;
      for (std::size_t d = 0; d < dim_size; ++d) {
        double p = std::exp(static_cast<double>(x->data[o * dim_size * inner + d * inner + inn]) -
                            max_val);
        out->data[o * dim_size * inner + d * inner + inn] = static_cast<T>(p);
        sum_exp += p;
      }

      for (std::size_t d = 0; d < dim_size; ++d) {
        out->data[o * dim_size * inner + d * inner + inn] = static_cast<T>(
            static_cast<double>(out->data[o * dim_size * inner + d * inner + inn]) / sum_exp);
      }
    }
  }

  if (rg) {
    out->backward_fn = [x, out, dim, outer, dim_size, inner]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t o = 0; o < outer; ++o) {
          for (std::size_t inn = 0; inn < inner; ++inn) {
            float sum_gy = 0.f;
            for (std::size_t d = 0; d < dim_size; ++d) {
              std::size_t idx = o * dim_size * inner + d * inner + inn;
              sum_gy += self->grad[idx] * static_cast<float>(out->data[idx]);
            }
            for (std::size_t d = 0; d < dim_size; ++d) {
              std::size_t idx = o * dim_size * inner + d * inner + inn;
              float y_val = static_cast<float>(out->data[idx]);
              x->grad[idx] += y_val * (self->grad[idx] - sum_gy);
            }
          }
        }
      }
    };
  }

  return out;
}

// Explicit template instantiations
template std::shared_ptr<Tensor<float>> relu(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> relu(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> relu(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> leaky_relu(const std::shared_ptr<Tensor<float>>&, float);
template std::shared_ptr<Tensor<double>> leaky_relu(const std::shared_ptr<Tensor<double>>&, double);
template std::shared_ptr<Tensor<int>> leaky_relu(const std::shared_ptr<Tensor<int>>&, int);

template std::shared_ptr<Tensor<float>> sigmoid(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> sigmoid(const std::shared_ptr<Tensor<double>>&);

template std::shared_ptr<Tensor<float>> tanh_op(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> tanh_op(const std::shared_ptr<Tensor<double>>&);

template std::shared_ptr<Tensor<float>> softmax(const std::shared_ptr<Tensor<float>>&, std::size_t);
template std::shared_ptr<Tensor<double>> softmax(const std::shared_ptr<Tensor<double>>&,
                                                 std::size_t);

}  // namespace scorch
