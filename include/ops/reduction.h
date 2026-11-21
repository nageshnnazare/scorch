#ifndef _AG_OPS_REDUCTION_H_
#define _AG_OPS_REDUCTION_H_

/**
 * @file reduction.h
 * @brief Differentiable reduction operations: sum, mean, max, sum_dim.
 */

#include <cstddef>
#include <memory>

#include "tensor.h"

namespace AG {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> reduce(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> max(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> sum_dim(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

}  // namespace AG

#endif  // !_AG_OPS_REDUCTION_H_
