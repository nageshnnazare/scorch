/**
 * @file matrix.cc
 * @brief Implementation of 2D matrix multiplication and transposition.
 *
 * @details
 * Implements differentiable matrix arithmetic:
 * - `matmul`: 2D matrix product C = A @ B with gradient accumulation for A and B.
 * - `transpose`: Matrix transposition with gradient transpose accumulation.
 */

#include "ops/matrix.h"

#include <stdexcept>

#include "ops/ops_common.h"

namespace scorch {

/**
 * @brief Performs 2D matrix multiplication with autograd backward support.
 * @param a First matrix of shape {M, K}.
 * @param b Second matrix of shape {K, N}.
 * @return Product matrix of shape {M, N}.
 * @throws std::invalid_argument If shapes are not rank-2 or inner dimensions do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a,
                                  const std::shared_ptr<Tensor<T>>& b) {
  if (a->shape.size() != 2 || b->shape.size() != 2) {
    throw std::invalid_argument("matmul: both operands must be rank-2");
  }

  const std::size_t m = a->shape[0];
  const std::size_t k = a->shape[1];
  if (b->shape[0] != k) {
    throw std::invalid_argument("matmul: inner dimensions do not match");
  }
  const std::size_t n = b->shape[1];

  bool rg = a->requires_grad || b->requires_grad;
  auto out = detail::make_result<T>({m, n}, rg, "matmul", {a, b});

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T acc = T(0);
      for (std::size_t t = 0; t < k; ++t) {
        acc += a->data[i * k + t] * b->data[t * n + j];
      }
      out->data[i * n + j] = acc;
    }
  }

  if (rg) {
    out->backward_fn = [a, b, m, k, n]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*a);
      detail::ensure_grad(*b);

      if (a->requires_grad) {
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t t = 0; t < k; ++t) {
            float acc = 0.f;
            for (std::size_t j = 0; j < n; ++j) {
              acc += self->grad[i * n + j] * static_cast<float>(b->data[t * n + j]);
            }
            a->grad[i * k + t] += acc;
          }
        }
      }

      if (b->requires_grad) {
        for (std::size_t t = 0; t < k; ++t) {
          for (std::size_t j = 0; j < n; ++j) {
            float acc = 0.f;
            for (std::size_t i = 0; i < m; ++i) {
              acc += static_cast<float>(a->data[i * k + t]) * self->grad[i * n + j];
            }
            b->grad[t * n + j] += acc;
          }
        }
      }
    };
  }

  return out;
}

/**
 * @brief Computes the transpose of a 2D matrix with backward gradient flow.
 * @param x Input 2D tensor of shape {M, N}.
 * @return Transposed tensor of shape {N, M}.
 * @throws std::invalid_argument If tensor is not rank-2.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& x) {
  if (x->shape.size() != 2) {
    throw std::invalid_argument("transpose: tensor must be rank-2");
  }

  const std::size_t rows = x->shape[0];
  const std::size_t cols = x->shape[1];

  bool rg = x->requires_grad;
  auto out = detail::make_result<T>({cols, rows}, rg, "transpose", {x});

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      out->data[j * rows + i] = x->data[i * cols + j];
    }
  }

  if (rg) {
    out->backward_fn = [x, rows, cols]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < rows; ++i) {
          for (std::size_t j = 0; j < cols; ++j) {
            x->grad[i * cols + j] += self->grad[j * rows + i];
          }
        }
      }
    };
  }

  return out;
}

// Explicit template instantiations
template std::shared_ptr<Tensor<float>> matmul(const std::shared_ptr<Tensor<float>>&,
                                               const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> matmul(const std::shared_ptr<Tensor<double>>&,
                                                const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> matmul(const std::shared_ptr<Tensor<int>>&,
                                             const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> transpose(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> transpose(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> transpose(const std::shared_ptr<Tensor<int>>&);

}  // namespace scorch
