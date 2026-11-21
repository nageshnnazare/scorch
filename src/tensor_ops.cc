/**
 * @file tensor_ops.cc
 * @brief Implementation of differentiable tensor operations with autograd backward passes.
 */

#include "tensor_ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace AG {

namespace {

/**
 * @brief Ensures gradient storage is allocated for a tensor that requires gradients.
 */
template <TensorScalar T>
void ensure_grad(Tensor<T>& t) {
  if (t.requires_grad && t.grad.size() != t.data.size()) {
    t.grad.assign(t.data.size(), 0.f);
  }
}

/**
 * @brief Creates a parent pointer handle for the computation graph.
 *
 * For intermediate nodes (which may be temporaries created in nested expressions
 * such as `(a + b) * a`), allocates on the heap to ensure the node survives
 * until backpropagation finishes.
 *
 * For leaf nodes (t.parents empty), uses a non-owning handle so the backward
 * pass writes gradients directly into the caller's leaf tensor on the stack.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> make_parent_handle(const Tensor<T>& t) {
  if (!t.parents.empty()) {
    return std::make_shared<Tensor<T>>(t);
  }
  return std::shared_ptr<Tensor<T>>(const_cast<Tensor<T>*>(&t), [](Tensor<T>*) {});
}

/**
 * @brief Constructs an operation result tensor linked to its computation graph parents.
 */
template <TensorScalar T>
Tensor<T> make_result(Shape shape, bool requires_grad, std::string op_name,
                      std::vector<std::shared_ptr<Tensor<T>>> parents) {
  auto out = Tensor<T>::zeros(std::move(shape), /*requires_grad=*/requires_grad);
  out.op_name = std::move(op_name);
  out.parents = std::move(parents);
  return out;
}

}  // namespace

// =============================================================================
// Element-wise Addition
// =============================================================================

template <TensorScalar T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape != b.shape) {
    throw std::invalid_argument("add: shape mismatch");
  }

  bool rg = a.requires_grad || b.requires_grad;
  auto a_ptr = make_parent_handle(a);
  auto b_ptr = make_parent_handle(b);

  auto out = make_result<T>(a.shape, rg, "add", {a_ptr, b_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = a.data[i] + b.data[i];
  }

  if (rg) {
    out.backward_fn = [a_ptr, b_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*a_ptr);
      ensure_grad(*b_ptr);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        if (a_ptr->requires_grad) {
          a_ptr->grad[i] += self->grad[i];
        }
        if (b_ptr->requires_grad) {
          b_ptr->grad[i] += self->grad[i];
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Subtraction
// =============================================================================

template <TensorScalar T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape != b.shape) {
    throw std::invalid_argument("sub: shape mismatch");
  }

  bool rg = a.requires_grad || b.requires_grad;
  auto a_ptr = make_parent_handle(a);
  auto b_ptr = make_parent_handle(b);

  auto out = make_result<T>(a.shape, rg, "sub", {a_ptr, b_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = a.data[i] - b.data[i];
  }

  if (rg) {
    out.backward_fn = [a_ptr, b_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*a_ptr);
      ensure_grad(*b_ptr);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        if (a_ptr->requires_grad) {
          a_ptr->grad[i] += self->grad[i];
        }
        if (b_ptr->requires_grad) {
          b_ptr->grad[i] += -self->grad[i];
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Multiplication (Hadamard Product)
// =============================================================================

template <TensorScalar T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape != b.shape) {
    throw std::invalid_argument("mul: shape mismatch");
  }

  bool rg = a.requires_grad || b.requires_grad;
  auto a_ptr = make_parent_handle(a);
  auto b_ptr = make_parent_handle(b);

  auto out = make_result<T>(a.shape, rg, "mul", {a_ptr, b_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = a.data[i] * b.data[i];
  }

  if (rg) {
    out.backward_fn = [a_ptr, b_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*a_ptr);
      ensure_grad(*b_ptr);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        if (a_ptr->requires_grad) {
          a_ptr->grad[i] += self->grad[i] * static_cast<float>(b_ptr->data[i]);
        }
        if (b_ptr->requires_grad) {
          b_ptr->grad[i] += self->grad[i] * static_cast<float>(a_ptr->data[i]);
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Division
// =============================================================================

template <TensorScalar T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape != b.shape) {
    throw std::invalid_argument("div: shape mismatch");
  }

  bool rg = a.requires_grad || b.requires_grad;
  auto a_ptr = make_parent_handle(a);
  auto b_ptr = make_parent_handle(b);

  auto out = make_result<T>(a.shape, rg, "div", {a_ptr, b_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = a.data[i] / b.data[i];
  }

  if (rg) {
    out.backward_fn = [a_ptr, b_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*a_ptr);
      ensure_grad(*b_ptr);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        float bv = static_cast<float>(b_ptr->data[i]);
        if (a_ptr->requires_grad) {
          a_ptr->grad[i] += self->grad[i] / bv;
        }
        if (b_ptr->requires_grad) {
          float av = static_cast<float>(a_ptr->data[i]);
          b_ptr->grad[i] += -self->grad[i] * av / (bv * bv);
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Negation
// =============================================================================

template <TensorScalar T>
Tensor<T> neg(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "neg", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = -x.data[i];
  }

  if (rg) {
    out.backward_fn = [x_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x_ptr->grad[i] += -self->grad[i];
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Power
// =============================================================================

template <TensorScalar T>
Tensor<T> pow(const Tensor<T>& x, T exponent) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "pow", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] =
        static_cast<T>(std::pow(static_cast<double>(x.data[i]), static_cast<double>(exponent)));
  }

  if (rg) {
    out.backward_fn = [x_ptr, exponent]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float xv = static_cast<float>(x_ptr->data[i]);
          float exp_f = static_cast<float>(exponent);
          x_ptr->grad[i] += self->grad[i] * exp_f * static_cast<float>(std::pow(xv, exp_f - 1.f));
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Exponential
// =============================================================================

template <TensorScalar T>
Tensor<T> exp(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "exp", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = static_cast<T>(std::exp(static_cast<double>(x.data[i])));
  }

  if (rg) {
    // Capture output values for backward: d/dx exp(x) = exp(x)
    auto out_data = out.data;
    out.backward_fn = [x_ptr, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x_ptr->grad[i] += self->grad[i] * static_cast<float>(out_data[i]);
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Element-wise Natural Logarithm
// =============================================================================

template <TensorScalar T>
Tensor<T> log(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "log", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = static_cast<T>(std::log(static_cast<double>(x.data[i])));
  }

  if (rg) {
    out.backward_fn = [x_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x_ptr->grad[i] += self->grad[i] / static_cast<float>(x_ptr->data[i]);
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Scalar Addition
// =============================================================================

template <TensorScalar T>
Tensor<T> add_scalar(const Tensor<T>& x, T scalar) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "add_scalar", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = x.data[i] + scalar;
  }

  if (rg) {
    out.backward_fn = [x_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x_ptr->grad[i] += self->grad[i];
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Scalar Multiplication
// =============================================================================

template <TensorScalar T>
Tensor<T> mul_scalar(const Tensor<T>& x, T scalar) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "mul_scalar", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = x.data[i] * scalar;
  }

  if (rg) {
    out.backward_fn = [x_ptr, scalar]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        float s = static_cast<float>(scalar);
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x_ptr->grad[i] += self->grad[i] * s;
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Matrix Multiplication
// =============================================================================

template <TensorScalar T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::invalid_argument("matmul: both operands must be rank-2");
  }

  const std::size_t m = a.shape[0];
  const std::size_t k = a.shape[1];
  if (b.shape[0] != k) {
    throw std::invalid_argument("matmul: inner dimensions do not match");
  }
  const std::size_t n = b.shape[1];

  bool rg = a.requires_grad || b.requires_grad;
  auto a_ptr = make_parent_handle(a);
  auto b_ptr = make_parent_handle(b);

  auto out = make_result<T>({m, n}, rg, "matmul", {a_ptr, b_ptr});

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T acc = T(0);
      for (std::size_t t = 0; t < k; ++t) {
        acc += a.data[i * k + t] * b.data[t * n + j];
      }
      out.data[i * n + j] = acc;
    }
  }

  if (rg) {
    out.backward_fn = [a_ptr, b_ptr, m, k, n]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*a_ptr);
      ensure_grad(*b_ptr);

      if (a_ptr->requires_grad) {
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t t = 0; t < k; ++t) {
            float acc = 0.f;
            for (std::size_t j = 0; j < n; ++j) {
              acc += self->grad[i * n + j] * static_cast<float>(b_ptr->data[t * n + j]);
            }
            a_ptr->grad[i * k + t] += acc;
          }
        }
      }

      if (b_ptr->requires_grad) {
        for (std::size_t t = 0; t < k; ++t) {
          for (std::size_t j = 0; j < n; ++j) {
            float acc = 0.f;
            for (std::size_t i = 0; i < m; ++i) {
              acc += static_cast<float>(a_ptr->data[i * k + t]) * self->grad[i * n + j];
            }
            b_ptr->grad[t * n + j] += acc;
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Rectified Linear Unit (ReLU)
// =============================================================================

template <TensorScalar T>
Tensor<T> relu(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "relu", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = (x.data[i] > T(0)) ? x.data[i] : T(0);
  }

  if (rg) {
    out.backward_fn = [x_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          if (x_ptr->data[i] > T(0)) {
            x_ptr->grad[i] += self->grad[i];
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Leaky ReLU
// =============================================================================

template <TensorScalar T>
Tensor<T> leaky_relu(const Tensor<T>& x, T alpha) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "leaky_relu", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = (x.data[i] > T(0)) ? x.data[i] : alpha * x.data[i];
  }

  if (rg) {
    out.backward_fn = [x_ptr, alpha]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        float a = static_cast<float>(alpha);
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          if (x_ptr->data[i] > T(0)) {
            x_ptr->grad[i] += self->grad[i];
          } else {
            x_ptr->grad[i] += self->grad[i] * a;
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Sigmoid Activation
// =============================================================================

template <TensorScalar T>
Tensor<T> sigmoid(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "sigmoid", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = static_cast<T>(1.0 / (1.0 + std::exp(-static_cast<double>(x.data[i]))));
  }

  if (rg) {
    auto out_data = out.data;
    out.backward_fn = [x_ptr, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float s = static_cast<float>(out_data[i]);
          x_ptr->grad[i] += self->grad[i] * s * (1.f - s);
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Tanh Activation
// =============================================================================

template <TensorScalar T>
Tensor<T> tanh_op(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "tanh", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = static_cast<T>(std::tanh(static_cast<double>(x.data[i])));
  }

  if (rg) {
    auto out_data = out.data;
    out.backward_fn = [x_ptr, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float t = static_cast<float>(out_data[i]);
          x_ptr->grad[i] += self->grad[i] * (1.f - t * t);
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Softmax (rank-2 tensors)
// =============================================================================

template <TensorScalar T>
Tensor<T> softmax(const Tensor<T>& x, std::size_t dim) {
  if (x.shape.size() != 2) {
    throw std::invalid_argument("softmax: only rank-2 tensors are supported");
  }
  if (dim > 1) {
    throw std::invalid_argument("softmax: dim must be 0 or 1 for rank-2 tensors");
  }

  const std::size_t rows = x.shape[0];
  const std::size_t cols = x.shape[1];
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>(x.shape, rg, "softmax", {x_ptr});

  if (dim == 1) {
    // Softmax along columns (each row is a distribution)
    for (std::size_t r = 0; r < rows; ++r) {
      // Numerical stability: subtract max
      T max_val = x.data[r * cols];
      for (std::size_t c = 1; c < cols; ++c) {
        if (x.data[r * cols + c] > max_val) max_val = x.data[r * cols + c];
      }
      T sum_exp = T(0);
      for (std::size_t c = 0; c < cols; ++c) {
        out.data[r * cols + c] =
            static_cast<T>(std::exp(static_cast<double>(x.data[r * cols + c] - max_val)));
        sum_exp += out.data[r * cols + c];
      }
      for (std::size_t c = 0; c < cols; ++c) {
        out.data[r * cols + c] /= sum_exp;
      }
    }
  } else {
    // dim == 0: softmax along rows (each column is a distribution)
    for (std::size_t c = 0; c < cols; ++c) {
      T max_val = x.data[c];
      for (std::size_t r = 1; r < rows; ++r) {
        if (x.data[r * cols + c] > max_val) max_val = x.data[r * cols + c];
      }
      T sum_exp = T(0);
      for (std::size_t r = 0; r < rows; ++r) {
        out.data[r * cols + c] =
            static_cast<T>(std::exp(static_cast<double>(x.data[r * cols + c] - max_val)));
        sum_exp += out.data[r * cols + c];
      }
      for (std::size_t r = 0; r < rows; ++r) {
        out.data[r * cols + c] /= sum_exp;
      }
    }
  }

  if (rg) {
    auto out_data = out.data;
    out.backward_fn = [x_ptr, out_data, rows, cols, dim]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        if (dim == 1) {
          for (std::size_t r = 0; r < rows; ++r) {
            // dot = sum_j(grad[j] * s[j])
            float dot = 0.f;
            for (std::size_t c = 0; c < cols; ++c) {
              dot += self->grad[r * cols + c] * static_cast<float>(out_data[r * cols + c]);
            }
            for (std::size_t c = 0; c < cols; ++c) {
              float s = static_cast<float>(out_data[r * cols + c]);
              x_ptr->grad[r * cols + c] += s * (self->grad[r * cols + c] - dot);
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
              x_ptr->grad[r * cols + c] += s * (self->grad[r * cols + c] - dot);
            }
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// 2D Transposition
// =============================================================================

template <TensorScalar T>
Tensor<T> transpose(const Tensor<T>& x) {
  if (x.shape.size() != 2) {
    throw std::invalid_argument("transpose: only rank-2 tensors are currently supported");
  }

  const std::size_t rows = x.shape[0];
  const std::size_t cols = x.shape[1];
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>({cols, rows}, rg, "transpose", {x_ptr});

  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      out.data[c * rows + r] = x.data[r * cols + c];
    }
  }

  if (rg) {
    out.backward_fn = [x_ptr, rows, cols]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t r = 0; r < rows; ++r) {
          for (std::size_t c = 0; c < cols; ++c) {
            x_ptr->grad[r * cols + c] += self->grad[c * rows + r];
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Reshape
// =============================================================================

template <TensorScalar T>
Tensor<T> reshape(const Tensor<T>& x, Shape new_shape) {
  if (numel(new_shape) != numel(x.shape)) {
    throw std::invalid_argument("reshape: numel mismatch");
  }

  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);
  Shape orig_shape = x.shape;

  auto out = make_result<T>(std::move(new_shape), rg, "reshape", {x_ptr});

  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = x.data[i];
  }

  if (rg) {
    out.backward_fn = [x_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x_ptr->grad[i] += self->grad[i];
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Flatten
// =============================================================================

template <TensorScalar T>
Tensor<T> flatten(const Tensor<T>& x) {
  return reshape(x, {numel(x.shape)});
}

// =============================================================================
// Squeeze (remove dimension of size 1)
// =============================================================================

template <TensorScalar T>
Tensor<T> squeeze(const Tensor<T>& x, std::size_t dim) {
  if (dim >= x.shape.size()) {
    throw std::invalid_argument("squeeze: dim out of range");
  }
  if (x.shape[dim] != 1) {
    throw std::invalid_argument("squeeze: dimension at dim is not size 1");
  }

  Shape new_shape;
  for (std::size_t i = 0; i < x.shape.size(); ++i) {
    if (i != dim) new_shape.push_back(x.shape[i]);
  }
  if (new_shape.empty()) new_shape.push_back(1);  // 0D -> scalar {1}

  return reshape(x, std::move(new_shape));
}

// =============================================================================
// Unsqueeze (insert dimension of size 1)
// =============================================================================

template <TensorScalar T>
Tensor<T> unsqueeze(const Tensor<T>& x, std::size_t dim) {
  if (dim > x.shape.size()) {
    throw std::invalid_argument("unsqueeze: dim out of range");
  }

  Shape new_shape = x.shape;
  new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(dim), 1);

  return reshape(x, std::move(new_shape));
}

// =============================================================================
// Concatenation
// =============================================================================

template <TensorScalar T>
Tensor<T> cat(const std::vector<const Tensor<T>*>& tensors, std::size_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: empty tensor list");
  }
  for (auto* t : tensors) {
    if (!t) throw std::invalid_argument("cat: null tensor pointer");
  }
  if (tensors[0]->shape.size() != 2) {
    throw std::invalid_argument("cat: only rank-2 tensors are supported");
  }
  if (dim > 1) {
    throw std::invalid_argument("cat: dim must be 0 or 1 for rank-2 tensors");
  }

  const auto& ref_shape = tensors[0]->shape;
  // Validate shapes match except along cat dimension
  for (std::size_t t = 1; t < tensors.size(); ++t) {
    if (tensors[t]->shape.size() != 2) {
      throw std::invalid_argument("cat: all tensors must be rank-2");
    }
    for (std::size_t d = 0; d < 2; ++d) {
      if (d != dim && tensors[t]->shape[d] != ref_shape[d]) {
        throw std::invalid_argument("cat: shape mismatch on non-cat dimension");
      }
    }
  }

  // Compute output shape
  std::size_t cat_size = 0;
  for (auto* t : tensors) cat_size += t->shape[dim];

  Shape out_shape = ref_shape;
  out_shape[dim] = cat_size;

  // Create parent handles
  bool rg = false;
  std::vector<std::shared_ptr<Tensor<T>>> parent_ptrs;
  for (auto* t : tensors) {
    rg = rg || t->requires_grad;
    parent_ptrs.push_back(make_parent_handle(*t));
  }

  auto out = make_result<T>(out_shape, rg, "cat", parent_ptrs);

  // Fill output data
  if (dim == 0) {
    // Concatenating rows: just append data sequentially
    std::size_t offset = 0;
    for (auto* t : tensors) {
      for (std::size_t i = 0; i < t->data.size(); ++i) {
        out.data[offset + i] = t->data[i];
      }
      offset += t->data.size();
    }
  } else {
    // dim == 1: interleave columns
    const std::size_t out_cols = out_shape[1];
    const std::size_t nrows = ref_shape[0];
    for (std::size_t r = 0; r < nrows; ++r) {
      std::size_t col_offset = 0;
      for (auto* t : tensors) {
        for (std::size_t c = 0; c < t->shape[1]; ++c) {
          out.data[r * out_cols + col_offset + c] = t->data[r * t->shape[1] + c];
        }
        col_offset += t->shape[1];
      }
    }
  }

  if (rg) {
    // Capture shapes for backward
    std::vector<Shape> input_shapes;
    for (auto* t : tensors) input_shapes.push_back(t->shape);

    out.backward_fn = [parent_ptrs, input_shapes, dim]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;

      for (auto& p : parent_ptrs) ensure_grad(*p);

      if (dim == 0) {
        std::size_t offset = 0;
        for (std::size_t t = 0; t < parent_ptrs.size(); ++t) {
          if (parent_ptrs[t]->requires_grad) {
            std::size_t n = parent_ptrs[t]->data.size();
            for (std::size_t i = 0; i < n; ++i) {
              parent_ptrs[t]->grad[i] += self->grad[offset + i];
            }
          }
          offset += parent_ptrs[t]->data.size();
        }
      } else {
        const std::size_t nrows = input_shapes[0][0];
        const std::size_t out_cols = self->grad.size() / nrows;
        for (std::size_t r = 0; r < nrows; ++r) {
          std::size_t col_offset = 0;
          for (std::size_t t = 0; t < parent_ptrs.size(); ++t) {
            if (parent_ptrs[t]->requires_grad) {
              for (std::size_t c = 0; c < input_shapes[t][1]; ++c) {
                parent_ptrs[t]->grad[r * input_shapes[t][1] + c] +=
                    self->grad[r * out_cols + col_offset + c];
              }
            }
            col_offset += input_shapes[t][1];
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Global Sum Reduction
// =============================================================================

template <TensorScalar T>
Tensor<T> reduce(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>({1}, rg, "reduce", {x_ptr});

  T sum = T(0);
  for (std::size_t i = 0; i < x.data.size(); ++i) {
    sum += x.data[i];
  }
  out.data[0] = sum;

  if (rg) {
    out.backward_fn = [x_ptr]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        float g = self->grad[0];
        for (std::size_t i = 0; i < x_ptr->grad.size(); ++i) {
          x_ptr->grad[i] += g;
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Global Mean Reduction
// =============================================================================

template <TensorScalar T>
Tensor<T> mean(const Tensor<T>& x) {
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);
  std::size_t n = x.data.size();

  auto out = make_result<T>({1}, rg, "mean", {x_ptr});

  T sum = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    sum += x.data[i];
  }
  out.data[0] = static_cast<T>(static_cast<double>(sum) / static_cast<double>(n));

  if (rg) {
    out.backward_fn = [x_ptr, n]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        float g = self->grad[0] / static_cast<float>(n);
        for (std::size_t i = 0; i < x_ptr->grad.size(); ++i) {
          x_ptr->grad[i] += g;
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Global Max Reduction
// =============================================================================

template <TensorScalar T>
Tensor<T> max(const Tensor<T>& x) {
  if (x.data.empty()) {
    throw std::invalid_argument("max: empty tensor");
  }

  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  auto out = make_result<T>({1}, rg, "max", {x_ptr});

  T max_val = x.data[0];
  for (std::size_t i = 1; i < x.data.size(); ++i) {
    if (x.data[i] > max_val) max_val = x.data[i];
  }
  out.data[0] = max_val;

  if (rg) {
    out.backward_fn = [x_ptr, max_val]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        float g = self->grad[0];
        for (std::size_t i = 0; i < x_ptr->grad.size(); ++i) {
          if (x_ptr->data[i] == max_val) {
            x_ptr->grad[i] += g;
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Sum Along Dimension (rank-2)
// =============================================================================

template <TensorScalar T>
Tensor<T> sum_dim(const Tensor<T>& x, std::size_t dim) {
  if (x.shape.size() != 2) {
    throw std::invalid_argument("sum_dim: only rank-2 tensors are supported");
  }
  if (dim > 1) {
    throw std::invalid_argument("sum_dim: dim must be 0 or 1");
  }

  const std::size_t rows = x.shape[0];
  const std::size_t cols = x.shape[1];
  bool rg = x.requires_grad;
  auto x_ptr = make_parent_handle(x);

  Shape out_shape;
  if (dim == 0) {
    out_shape = {1, cols};
  } else {
    out_shape = {rows, 1};
  }

  auto out = make_result<T>(out_shape, rg, "sum_dim", {x_ptr});

  if (dim == 0) {
    // Sum down rows -> shape {1, cols}
    for (std::size_t c = 0; c < cols; ++c) {
      T sum = T(0);
      for (std::size_t r = 0; r < rows; ++r) {
        sum += x.data[r * cols + c];
      }
      out.data[c] = sum;
    }
  } else {
    // Sum across columns -> shape {rows, 1}
    for (std::size_t r = 0; r < rows; ++r) {
      T sum = T(0);
      for (std::size_t c = 0; c < cols; ++c) {
        sum += x.data[r * cols + c];
      }
      out.data[r] = sum;
    }
  }

  if (rg) {
    out.backward_fn = [x_ptr, rows, cols, dim]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*x_ptr);

      if (x_ptr->requires_grad) {
        if (dim == 0) {
          // Broadcast grad {1, cols} -> {rows, cols}
          for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
              x_ptr->grad[r * cols + c] += self->grad[c];
            }
          }
        } else {
          // Broadcast grad {rows, 1} -> {rows, cols}
          for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
              x_ptr->grad[r * cols + c] += self->grad[r];
            }
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Mean Squared Error Loss
// =============================================================================

template <TensorScalar T>
Tensor<T> mse_loss(const Tensor<T>& pred, const Tensor<T>& target) {
  if (pred.shape != target.shape) {
    throw std::invalid_argument("mse_loss: shape mismatch");
  }

  bool rg = pred.requires_grad || target.requires_grad;
  auto pred_ptr = make_parent_handle(pred);
  auto target_ptr = make_parent_handle(target);
  std::size_t n = pred.data.size();

  auto out = make_result<T>({1}, rg, "mse_loss", {pred_ptr, target_ptr});

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(pred.data[i]) - static_cast<double>(target.data[i]);
    sum += diff * diff;
  }
  out.data[0] = static_cast<T>(sum / static_cast<double>(n));

  if (rg) {
    out.backward_fn = [pred_ptr, target_ptr, n]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*pred_ptr);
      ensure_grad(*target_ptr);

      float g = self->grad[0];
      float inv_n = 1.f / static_cast<float>(n);

      for (std::size_t i = 0; i < n; ++i) {
        float diff =
            static_cast<float>(pred_ptr->data[i]) - static_cast<float>(target_ptr->data[i]);
        if (pred_ptr->requires_grad) {
          pred_ptr->grad[i] += g * 2.f * diff * inv_n;
        }
        if (target_ptr->requires_grad) {
          target_ptr->grad[i] += g * -2.f * diff * inv_n;
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Cross-Entropy Loss (with integer class targets)
// =============================================================================

template <TensorScalar T>
Tensor<T> cross_entropy_loss(const Tensor<T>& logits, const Tensor<int>& targets) {
  if (logits.shape.size() != 2) {
    throw std::invalid_argument("cross_entropy_loss: logits must be rank-2 {N, C}");
  }
  if (targets.shape.size() != 1 || targets.shape[0] != logits.shape[0]) {
    throw std::invalid_argument("cross_entropy_loss: targets must have shape {N}");
  }

  const std::size_t batch = logits.shape[0];
  const std::size_t classes = logits.shape[1];
  bool rg = logits.requires_grad;
  auto logits_ptr = make_parent_handle(logits);

  auto out = make_result<T>({1}, rg, "cross_entropy_loss", {logits_ptr});

  // Forward: loss = mean_i( -logits[i][t_i] + log(sum_j(exp(logits[i][j]))) )
  // Use log-sum-exp trick for numerical stability
  double total_loss = 0.0;
  for (std::size_t i = 0; i < batch; ++i) {
    int ti = targets.data[i];
    // Find max for numerical stability
    double max_val = static_cast<double>(logits.data[i * classes]);
    for (std::size_t j = 1; j < classes; ++j) {
      double v = static_cast<double>(logits.data[i * classes + j]);
      if (v > max_val) max_val = v;
    }
    double log_sum_exp = 0.0;
    for (std::size_t j = 0; j < classes; ++j) {
      log_sum_exp += std::exp(static_cast<double>(logits.data[i * classes + j]) - max_val);
    }
    log_sum_exp = max_val + std::log(log_sum_exp);
    total_loss += -static_cast<double>(logits.data[i * classes + ti]) + log_sum_exp;
  }
  out.data[0] = static_cast<T>(total_loss / static_cast<double>(batch));

  if (rg) {
    // Capture targets data
    auto target_data = targets.data;
    out.backward_fn = [logits_ptr, target_data, batch, classes]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      ensure_grad(*logits_ptr);

      if (logits_ptr->requires_grad) {
        float g = self->grad[0];
        float inv_n = 1.f / static_cast<float>(batch);

        for (std::size_t i = 0; i < batch; ++i) {
          // Compute softmax for this sample
          double max_val = static_cast<double>(logits_ptr->data[i * classes]);
          for (std::size_t j = 1; j < classes; ++j) {
            double v = static_cast<double>(logits_ptr->data[i * classes + j]);
            if (v > max_val) max_val = v;
          }
          double sum_exp = 0.0;
          std::vector<float> sm(classes);
          for (std::size_t j = 0; j < classes; ++j) {
            sm[j] = static_cast<float>(
                std::exp(static_cast<double>(logits_ptr->data[i * classes + j]) - max_val));
            sum_exp += sm[j];
          }
          for (std::size_t j = 0; j < classes; ++j) {
            sm[j] /= static_cast<float>(sum_exp);
          }

          int ti = target_data[i];
          for (std::size_t j = 0; j < classes; ++j) {
            float indicator = (static_cast<int>(j) == ti) ? 1.f : 0.f;
            logits_ptr->grad[i * classes + j] += g * (sm[j] - indicator) * inv_n;
          }
        }
      }
    };
  }

  return out;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

// --- Arithmetic ---
template Tensor<float> add(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> add(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> add(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> sub(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> sub(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> sub(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> mul(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> mul(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> mul(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> div(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> div(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> div(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> neg(const Tensor<float>&);
template Tensor<double> neg(const Tensor<double>&);
template Tensor<int> neg(const Tensor<int>&);

template Tensor<float> pow(const Tensor<float>&, float);
template Tensor<double> pow(const Tensor<double>&, double);
template Tensor<int> pow(const Tensor<int>&, int);

template Tensor<float> exp(const Tensor<float>&);
template Tensor<double> exp(const Tensor<double>&);
template Tensor<int> exp(const Tensor<int>&);

template Tensor<float> log(const Tensor<float>&);
template Tensor<double> log(const Tensor<double>&);
template Tensor<int> log(const Tensor<int>&);

// --- Scalar-Tensor ---
template Tensor<float> add_scalar(const Tensor<float>&, float);
template Tensor<double> add_scalar(const Tensor<double>&, double);
template Tensor<int> add_scalar(const Tensor<int>&, int);

template Tensor<float> mul_scalar(const Tensor<float>&, float);
template Tensor<double> mul_scalar(const Tensor<double>&, double);
template Tensor<int> mul_scalar(const Tensor<int>&, int);

// --- Matrix ---
template Tensor<float> matmul(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> matmul(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> matmul(const Tensor<int>&, const Tensor<int>&);

// --- Activations ---
template Tensor<float> relu(const Tensor<float>&);
template Tensor<double> relu(const Tensor<double>&);
template Tensor<int> relu(const Tensor<int>&);

template Tensor<float> leaky_relu(const Tensor<float>&, float);
template Tensor<double> leaky_relu(const Tensor<double>&, double);
template Tensor<int> leaky_relu(const Tensor<int>&, int);

template Tensor<float> sigmoid(const Tensor<float>&);
template Tensor<double> sigmoid(const Tensor<double>&);
template Tensor<int> sigmoid(const Tensor<int>&);

template Tensor<float> tanh_op(const Tensor<float>&);
template Tensor<double> tanh_op(const Tensor<double>&);
template Tensor<int> tanh_op(const Tensor<int>&);

template Tensor<float> softmax(const Tensor<float>&, std::size_t);
template Tensor<double> softmax(const Tensor<double>&, std::size_t);
template Tensor<int> softmax(const Tensor<int>&, std::size_t);

// --- Shape / View ---
template Tensor<float> transpose(const Tensor<float>&);
template Tensor<double> transpose(const Tensor<double>&);
template Tensor<int> transpose(const Tensor<int>&);

template Tensor<float> reshape(const Tensor<float>&, Shape);
template Tensor<double> reshape(const Tensor<double>&, Shape);
template Tensor<int> reshape(const Tensor<int>&, Shape);

template Tensor<float> flatten(const Tensor<float>&);
template Tensor<double> flatten(const Tensor<double>&);
template Tensor<int> flatten(const Tensor<int>&);

template Tensor<float> squeeze(const Tensor<float>&, std::size_t);
template Tensor<double> squeeze(const Tensor<double>&, std::size_t);
template Tensor<int> squeeze(const Tensor<int>&, std::size_t);

template Tensor<float> unsqueeze(const Tensor<float>&, std::size_t);
template Tensor<double> unsqueeze(const Tensor<double>&, std::size_t);
template Tensor<int> unsqueeze(const Tensor<int>&, std::size_t);

template Tensor<float> cat(const std::vector<const Tensor<float>*>&, std::size_t);
template Tensor<double> cat(const std::vector<const Tensor<double>*>&, std::size_t);
template Tensor<int> cat(const std::vector<const Tensor<int>*>&, std::size_t);

// --- Reductions ---
template Tensor<float> reduce(const Tensor<float>&);
template Tensor<double> reduce(const Tensor<double>&);
template Tensor<int> reduce(const Tensor<int>&);

template Tensor<float> mean(const Tensor<float>&);
template Tensor<double> mean(const Tensor<double>&);
template Tensor<int> mean(const Tensor<int>&);

template Tensor<float> max(const Tensor<float>&);
template Tensor<double> max(const Tensor<double>&);
template Tensor<int> max(const Tensor<int>&);

template Tensor<float> sum_dim(const Tensor<float>&, std::size_t);
template Tensor<double> sum_dim(const Tensor<double>&, std::size_t);
template Tensor<int> sum_dim(const Tensor<int>&, std::size_t);

// --- Loss Functions ---
template Tensor<float> mse_loss(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> mse_loss(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> mse_loss(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> cross_entropy_loss(const Tensor<float>&, const Tensor<int>&);
template Tensor<double> cross_entropy_loss(const Tensor<double>&, const Tensor<int>&);

}  // namespace AG
