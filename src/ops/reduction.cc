/**
 * @file reduction.cc
 * @brief Implementation of reduction operations: sum, mean, max, sum_dim.
 *
 * @details
 * Defines differentiable tensor reductions:
 * - `reduce`: Full sum across all elements.
 * - `mean`: Global arithmetic mean.
 * - `max`: Global maximum element extraction.
 * - `sum_dim`: Dimension-wise sum reduction along a specified axis.
 */

#include "ops/reduction.h"

#include <stdexcept>

#include "ops/ops_common.h"

namespace scorch {

/**
 * @brief Computes the sum of all elements in the tensor.
 * @param x Input tensor.
 * @return 1-element scalar sum tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reduce(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>({1}, rg, "reduce", {x});

  T sum = T(0);
  for (std::size_t i = 0; i < x->data.size(); ++i) {
    sum += x->data[i];
  }
  out->data[0] = sum;

  if (rg) {
    out->backward_fn = [x]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        float g = self->grad[0];
        for (std::size_t i = 0; i < x->grad.size(); ++i) {
          x->grad[i] += g;
        }
      }
    };
  }

  return out;
}

/**
 * @brief Computes the arithmetic mean of all elements in the tensor.
 * @param x Input tensor.
 * @return 1-element scalar mean tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  std::size_t n = x->data.size();
  auto out = detail::make_result<T>({1}, rg, "mean", {x});

  T sum = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    sum += x->data[i];
  }
  out->data[0] = static_cast<T>(static_cast<double>(sum) / static_cast<double>(n));

  if (rg) {
    out->backward_fn = [x, n]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        float g = self->grad[0] / static_cast<float>(n);
        for (std::size_t i = 0; i < x->grad.size(); ++i) {
          x->grad[i] += g;
        }
      }
    };
  }

  return out;
}

/**
 * @brief Computes the maximum value across all elements in the tensor.
 * @param x Input tensor.
 * @return 1-element scalar maximum tensor.
 * @throws std::invalid_argument If tensor is empty.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> max(const std::shared_ptr<Tensor<T>>& x) {
  if (x->data.empty()) {
    throw std::invalid_argument("max: empty tensor");
  }

  bool rg = x->requires_grad;
  auto out = detail::make_result<T>({1}, rg, "max", {x});

  T max_val = x->data[0];
  for (std::size_t i = 1; i < x->data.size(); ++i) {
    if (x->data[i] > max_val) max_val = x->data[i];
  }
  out->data[0] = max_val;

  if (rg) {
    out->backward_fn = [x, max_val]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        float g = self->grad[0];
        for (std::size_t i = 0; i < x->grad.size(); ++i) {
          if (x->data[i] == max_val) {
            x->grad[i] += g;
          }
        }
      }
    };
  }

  return out;
}

/**
 * @brief Computes the sum along a specified dimension for rank-2 tensors.
 * @param x Input 2D tensor.
 * @param dim Dimension to reduce (0 for column sum, 1 for row sum).
 * @return Reduced 2D tensor of shape {1, cols} or {rows, 1}.
 * @throws std::invalid_argument If x is not rank-2 or dim > 1.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sum_dim(const std::shared_ptr<Tensor<T>>& x, std::size_t dim) {
  if (x->shape.size() != 2) {
    throw std::invalid_argument("sum_dim: only rank-2 tensors are supported");
  }
  if (dim > 1) {
    throw std::invalid_argument("sum_dim: dim must be 0 or 1");
  }

  const std::size_t rows = x->shape[0];
  const std::size_t cols = x->shape[1];
  bool rg = x->requires_grad;

  Shape out_shape;
  if (dim == 0) {
    out_shape = {1, cols};
  } else {
    out_shape = {rows, 1};
  }

  auto out = detail::make_result<T>(out_shape, rg, "sum_dim", {x});

  if (dim == 0) {
    for (std::size_t c = 0; c < cols; ++c) {
      T sum = T(0);
      for (std::size_t r = 0; r < rows; ++r) {
        sum += x->data[r * cols + c];
      }
      out->data[c] = sum;
    }
  } else {
    for (std::size_t r = 0; r < rows; ++r) {
      T sum = T(0);
      for (std::size_t c = 0; c < cols; ++c) {
        sum += x->data[r * cols + c];
      }
      out->data[r] = sum;
    }
  }

  if (rg) {
    out->backward_fn = [x, rows, cols, dim]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        if (dim == 0) {
          for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
              x->grad[r * cols + c] += self->grad[c];
            }
          }
        } else {
          for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
              x->grad[r * cols + c] += self->grad[r];
            }
          }
        }
      }
    };
  }

  return out;
}

// Explicit template instantiations
template std::shared_ptr<Tensor<float>> reduce(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> reduce(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> reduce(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> mean(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> mean(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> mean(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> max(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> max(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> max(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> sum_dim(const std::shared_ptr<Tensor<float>>&, std::size_t);
template std::shared_ptr<Tensor<double>> sum_dim(const std::shared_ptr<Tensor<double>>&,
                                                 std::size_t);
template std::shared_ptr<Tensor<int>> sum_dim(const std::shared_ptr<Tensor<int>>&, std::size_t);

}  // namespace scorch
