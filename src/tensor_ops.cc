/**
 * @file tensor_ops.cc
 * @brief Implementation of differentiable tensor operations with autograd backward passes.
 */

#include "tensor_ops.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace AG {

namespace {

/**
 * @brief Ensures gradient storage is allocated for a tensor that requires gradients.
 * Reuses shape element count from numel() via t.data.size().
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
 *
 * Reuses Tensor<T>::zeros from tensor.cc for buffer allocation.
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

// -----------------------------------------------------------------------------
// Element-wise Addition
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Element-wise Multiplication (Hadamard Product)
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Matrix Multiplication
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Rectified Linear Unit (ReLU)
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// 2D Transposition
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Global Sum Reduction
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Explicit Template Instantiations
// -----------------------------------------------------------------------------

template Tensor<float> add(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> add(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> add(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> mul(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> mul(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> mul(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> matmul(const Tensor<float>&, const Tensor<float>&);
template Tensor<double> matmul(const Tensor<double>&, const Tensor<double>&);
template Tensor<int> matmul(const Tensor<int>&, const Tensor<int>&);

template Tensor<float> relu(const Tensor<float>&);
template Tensor<double> relu(const Tensor<double>&);
template Tensor<int> relu(const Tensor<int>&);

template Tensor<float> transpose(const Tensor<float>&);
template Tensor<double> transpose(const Tensor<double>&);
template Tensor<int> transpose(const Tensor<int>&);

template Tensor<float> reduce(const Tensor<float>&);
template Tensor<double> reduce(const Tensor<double>&);
template Tensor<int> reduce(const Tensor<int>&);

}  // namespace AG
