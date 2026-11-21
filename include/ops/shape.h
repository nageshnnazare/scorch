#ifndef _SCORCH_OPS_SHAPE_H_
#define _SCORCH_OPS_SHAPE_H_

/**
 * @file shape.h
 * @brief Differentiable tensor shape and view manipulations.
 *
 * @details
 * Provides functions for transforming tensor dimensions while preserving data
 * and gradient continuity:
 * - `reshape(x, shape)`: View or reallocate tensor with a new shape of identical element count.
 * - `flatten(x)`: Flatten all dimensions into a 1-D vector.
 * - `squeeze(x, dim)`: Remove a dimension of size 1.
 * - `unsqueeze(x, dim)`: Insert a dimension of size 1 at the specified axis.
 * - `cat(tensors, dim)`: Concatenate multiple tensors along an existing dimension.
 *
 * @ingroup ops
 */

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "tensor.h"

namespace scorch {

/**
 * @brief Reshapes a tensor to a new shape with the same total number of elements.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor to reshape.
 * @param new_shape Target dimensions (must have the same `numel`).
 * @return Reshaped tensor containing identical data elements.
 * @throws std::invalid_argument If the total number of elements does not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reshape(const std::shared_ptr<Tensor<T>>& x, Shape new_shape);

/**
 * @brief Flattens a tensor into a 1-D tensor of shape `{numel(x)}`.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor of any rank.
 * @return 1-D flattened tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> flatten(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Removes a dimension of size 1 at the specified axis.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param dim Index of the dimension to remove (must satisfy `x->shape[dim] == 1`).
 * @return Tensor with dimension `dim` removed.
 * @throws std::out_of_range If `dim >= x->shape.size()`.
 * @throws std::invalid_argument If `x->shape[dim] != 1`.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> squeeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

/**
 * @brief Inserts a singleton dimension (size 1) at the specified axis.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param dim Index at which to insert dimension 1 (`0 <= dim <= rank`).
 * @return Tensor with rank increased by 1 having size 1 at index `dim`.
 * @throws std::out_of_range If `dim > x->shape.size()`.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> unsqueeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

/**
 * @brief Concatenates a sequence of tensors along an existing dimension.
 *
 * @details
 * All tensors must have the same shape except along the concatenation axis `dim`.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param tensors Vector of tensor shared pointers to concatenate.
 * @param dim Dimension along which to concatenate.
 * @return Concatenated tensor.
 * @throws std::invalid_argument If tensors is empty or shapes do not match along non-concatenated
 * axes.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cat(const std::vector<std::shared_ptr<Tensor<T>>>& tensors,
                               std::size_t dim);

/**
 * @brief Convenience initializer_list overload for tensor concatenation.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param tensors Initializer list of tensor shared pointers.
 * @param dim Dimension along which to concatenate.
 * @return Concatenated tensor.
 */
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> cat(std::initializer_list<std::shared_ptr<Tensor<T>>> tensors,
                                      std::size_t dim) {
  return cat(std::vector<std::shared_ptr<Tensor<T>>>(tensors.begin(), tensors.end()), dim);
}

}  // namespace scorch

#endif  // !_SCORCH_OPS_SHAPE_H_
