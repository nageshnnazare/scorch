#ifndef _SCORCH_OPS_ARITHMETIC_H_
#define _SCORCH_OPS_ARITHMETIC_H_

/**
 * @file arithmetic.h
 * @brief Differentiable element-wise arithmetic and scalar operations.
 */

#include <memory>

#include "tensor.h"

namespace scorch {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> sub(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> div(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> neg(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> pow(const std::shared_ptr<Tensor<T>>& x, T exponent);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> exp(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> add_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

}  // namespace scorch

#endif  // !_SCORCH_OPS_ARITHMETIC_H_
