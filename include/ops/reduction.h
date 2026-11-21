#ifndef _SCORCH_OPS_REDUCTION_H_
#define _SCORCH_OPS_REDUCTION_H_

/**
 * @file reduction.h
 * @brief Differentiable reduction operations: sum, mean, max, and dimension-wise sum.
 *
 * @details
 * Provides autograd-aware reduction operations that reduce multidimensional tensors
 * to scalars or smaller dimensional projections.
 *
 * @ingroup ops
 */

#include <cstddef>
#include <memory>

#include "tensor.h"

namespace scorch {

/**
 * @brief Computes the sum of all elements in the tensor.
 *
 * @details
 * Reduces tensor to a 1-D scalar `{1}`.
 * Backward: Incoming gradient is broadcast to all input elements (\f$\partial L / \partial x_i =
 * \partial L / \partial y\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Scalar tensor containing the sum of all elements.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reduce(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Computes the arithmetic mean of all elements in the tensor.
 *
 * @details
 * Reduces tensor to a 1-D scalar `{1}`.
 * Backward: Incoming gradient divided by `numel(x)` is broadcast to all input elements.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Scalar tensor containing the mean value.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Computes the maximum value across all elements in the tensor.
 *
 * @details
 * Reduces tensor to a 1-D scalar `{1}`.
 * Backward: Incoming gradient is routed exclusively to the element with the maximum value.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Scalar tensor containing the maximum element.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> max(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Computes the sum along a specified dimension, reducing that axis.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param dim Axis along which elements are summed.
 * @return Reduced tensor with dimension `dim` removed (or dimension 1 if rank 1).
 * @throws std::out_of_range If `dim >= x->shape.size()`.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sum_dim(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

}  // namespace scorch

#endif  // !_SCORCH_OPS_REDUCTION_H_
