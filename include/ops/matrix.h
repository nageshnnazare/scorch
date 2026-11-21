#ifndef _AG_OPS_MATRIX_H_
#define _AG_OPS_MATRIX_H_

/**
 * @file matrix.h
 * @brief Differentiable 2D matrix multiplication and transposition.
 */

#include <memory>

#include "tensor.h"

namespace AG {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a,
                                  const std::shared_ptr<Tensor<T>>& b);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& x);

}  // namespace AG

#endif  // !_AG_OPS_MATRIX_H_
