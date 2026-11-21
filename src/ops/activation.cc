/**
 * @file activation.cc
 * @brief Implementation of activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax.
 */

#include "ops/activation.h"

#include <cmath>
#include <stdexcept>

#include "ops/ops_common.h"

namespace scorch {

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
        float a = static_cast<float>(alpha);
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          if (x->data[i] > T(0)) {
            x->grad[i] += self->grad[i];
          } else {
            x->grad[i] += self->grad[i] * a;
          }
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "sigmoid", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(1.0 / (1.0 + std::exp(-static_cast<double>(x->data[i]))));
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float s = static_cast<float>(out_data[i]);
          x->grad[i] += self->grad[i] * s * (1.f - s);
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> tanh_op(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "tanh", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::tanh(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float t = static_cast<float>(out_data[i]);
          x->grad[i] += self->grad[i] * (1.f - t * t);
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& x, std::size_t dim) {
  if (x->shape.size() != 2) {
    throw std::invalid_argument("softmax: only rank-2 tensors are supported");
  }
  if (dim > 1) {
    throw std::invalid_argument("softmax: dim must be 0 or 1 for rank-2 tensors");
  }

  const std::size_t rows = x->shape[0];
  const std::size_t cols = x->shape[1];
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "softmax", {x});

  if (dim == 1) {
    for (std::size_t r = 0; r < rows; ++r) {
      T max_val = x->data[r * cols];
      for (std::size_t c = 1; c < cols; ++c) {
        if (x->data[r * cols + c] > max_val) max_val = x->data[r * cols + c];
      }
      T sum_exp = T(0);
      for (std::size_t c = 0; c < cols; ++c) {
        out->data[r * cols + c] =
            static_cast<T>(std::exp(static_cast<double>(x->data[r * cols + c] - max_val)));
        sum_exp += out->data[r * cols + c];
      }
      for (std::size_t c = 0; c < cols; ++c) {
        out->data[r * cols + c] /= sum_exp;
      }
    }
  } else {
    for (std::size_t c = 0; c < cols; ++c) {
      T max_val = x->data[c];
      for (std::size_t r = 1; r < rows; ++r) {
        if (x->data[r * cols + c] > max_val) max_val = x->data[r * cols + c];
      }
      T sum_exp = T(0);
      for (std::size_t r = 0; r < rows; ++r) {
        out->data[r * cols + c] =
            static_cast<T>(std::exp(static_cast<double>(x->data[r * cols + c] - max_val)));
        sum_exp += out->data[r * cols + c];
      }
      for (std::size_t r = 0; r < rows; ++r) {
        out->data[r * cols + c] /= sum_exp;
      }
    }
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data, rows, cols, dim]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        if (dim == 1) {
          for (std::size_t r = 0; r < rows; ++r) {
            float dot = 0.f;
            for (std::size_t c = 0; c < cols; ++c) {
              dot += self->grad[r * cols + c] * static_cast<float>(out_data[r * cols + c]);
            }
            for (std::size_t c = 0; c < cols; ++c) {
              float s = static_cast<float>(out_data[r * cols + c]);
              x->grad[r * cols + c] += s * (self->grad[r * cols + c] - dot);
            }
          }
        } else {
          for (std::size_t c = 0; c < cols; ++c) {
            float dot = 0.f;
            for (std::size_t r = 0; r < rows; ++r) {
              dot += self->grad[r * cols + c] * static_cast<float>(out_data[r * cols + c]);
            }
            for (std::size_t r = 0; r < rows; ++r) {
              float s = static_cast<float>(out_data[r * cols + c]);
              x->grad[r * cols + c] += s * (self->grad[r * cols + c] - dot);
            }
          }
        }
      }
    };
  }

  return out;
}

// Explicit Instantiations
template std::shared_ptr<Tensor<float>> relu(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> relu(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> relu(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> leaky_relu(const std::shared_ptr<Tensor<float>>&, float);
template std::shared_ptr<Tensor<double>> leaky_relu(const std::shared_ptr<Tensor<double>>&, double);
template std::shared_ptr<Tensor<int>> leaky_relu(const std::shared_ptr<Tensor<int>>&, int);

template std::shared_ptr<Tensor<float>> sigmoid(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> sigmoid(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> sigmoid(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> tanh_op(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> tanh_op(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> tanh_op(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> softmax(const std::shared_ptr<Tensor<float>>&, std::size_t);
template std::shared_ptr<Tensor<double>> softmax(const std::shared_ptr<Tensor<double>>&,
                                                 std::size_t);
template std::shared_ptr<Tensor<int>> softmax(const std::shared_ptr<Tensor<int>>&, std::size_t);

}  // namespace scorch
