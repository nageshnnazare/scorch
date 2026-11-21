#ifndef _SCORCH_OPS_TRIGONOMETRIC_H_
#define _SCORCH_OPS_TRIGONOMETRIC_H_

/**
 * @file trigonometric.h
 * @brief Differentiable trigonometric and advanced mathematical functions.
 */

#include <memory>

#include "ops/activation.h"
#include "ops/arithmetic.h"
#include "tensor.h"

namespace scorch {

// Trigonometric functions
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sin(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> cos(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> tan(const std::shared_ptr<Tensor<T>>& x);

// Inverse trigonometric functions
template <TensorScalar T>
std::shared_ptr<Tensor<T>> asin(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> acos(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> atan(const std::shared_ptr<Tensor<T>>& x);

// Hyperbolic functions
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sinh(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> cosh(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> tanh(const std::shared_ptr<Tensor<T>>& x) {
  return tanh_op(x);
}

// Logarithmic functions
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> loge(const std::shared_ptr<Tensor<T>>& x) {
  return log(x);
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> log2(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> log10(const std::shared_ptr<Tensor<T>>& x);

// Roots and absolute value
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sqrt(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> rsqrt(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> abs(const std::shared_ptr<Tensor<T>>& x);

}  // namespace scorch

#endif  // !_SCORCH_OPS_TRIGONOMETRIC_H_
