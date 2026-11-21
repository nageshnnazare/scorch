#ifndef _SCORCH_OPS_MATRIX_H_
#define _SCORCH_OPS_MATRIX_H_

/**
 * @file matrix.h
 * @brief Differentiable 2D matrix multiplication and transposition.
 */

#include <memory>

#include "tensor.h"

namespace scorch {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a,
                                  const std::shared_ptr<Tensor<T>>& b);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& x);

}  // namespace scorch

#endif  // !_SCORCH_OPS_MATRIX_H_
