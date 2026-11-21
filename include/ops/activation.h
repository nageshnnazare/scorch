#ifndef _AG_OPS_ACTIVATION_H_
#define _AG_OPS_ACTIVATION_H_

/**
 * @file activation.h
 * @brief Differentiable activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax.
 */

#include <cstddef>
#include <memory>

#include "tensor.h"

namespace AG {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> leaky_relu(const std::shared_ptr<Tensor<T>>& x, T alpha);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> tanh_op(const std::shared_ptr<Tensor<T>>& x);

template <TensorScalar T>
std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

}  // namespace AG

#endif  // !_AG_OPS_ACTIVATION_H_
