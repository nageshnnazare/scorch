#ifndef _SCORCH_OPS_SHAPE_H_
#define _SCORCH_OPS_SHAPE_H_

/**
 * @file shape.h
 * @brief Differentiable tensor shape and view manipulations: reshape, flatten, squeeze, unsqueeze,
 * cat.
 */

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "tensor.h"

namespace scorch {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> reshape(const std::shared_ptr<Tensor<T>>& x, Shape new_shape);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> flatten(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> squeeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> unsqueeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> cat(const std::vector<std::shared_ptr<Tensor<T>>>& tensors,
                               std::size_t dim);

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> cat(std::initializer_list<std::shared_ptr<Tensor<T>>> tensors,
                                      std::size_t dim) {
  return cat(std::vector<std::shared_ptr<Tensor<T>>>(tensors.begin(), tensors.end()), dim);
}

}  // namespace scorch

#endif  // !_SCORCH_OPS_SHAPE_H_
