#ifndef _SCORCH_OPS_ACTIVATION_H_
#define _SCORCH_OPS_ACTIVATION_H_

/**
 * @file activation.h
 * @brief Differentiable activation functions.
 *
 * @details
 * Provides free-function autograd-aware activation functions:
 * - `relu`: \f$\max(0, x)\f$
 * - `leaky_relu`: \f$\max(\alpha x, x)\f$
 * - `sigmoid`: \f$\frac{1}{1 + e^{-x}}\f$
 * - `tanh_op`: \f$\tanh(x)\f$
 * - `softmax`: normalized exponent along specified axis
 *
 * @ingroup ops
 */

#include <cstddef>
#include <memory>

#include "tensor.h"

namespace scorch {

/**
 * @brief Rectified Linear Unit activation applied element-wise.
 *
 * @details
 * Computes \f$y = \max(0, x)\f$.
 * Backward: \f$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x >
 * 0]\f$.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Tensor of identical shape with negative values clamped to 0.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Leaky Rectified Linear Unit activation applied element-wise.
 *
 * @details
 * Computes \f$y = \max(\alpha x, x)\f$ where \f$\alpha\f$ is negative slope.
 * Backward: \f$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (\mathbf{1}[x >
 * 0] + \alpha \mathbf{1}[x \le 0])\f$.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param alpha Negative slope multiplier.
 * @return Tensor of identical shape with negative values scaled by `alpha`.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> leaky_relu(const std::shared_ptr<Tensor<T>>& x, T alpha);

/**
 * @brief Sigmoid (logistic) function applied element-wise.
 *
 * @details
 * Computes \f$y = \frac{1}{1 + e^{-x}}\f$.
 * Backward: \f$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot y(1 - y)\f$.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Tensor of identical shape with values mapped to \f$(0, 1)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Hyperbolic Tangent function applied element-wise.
 *
 * @details
 * Computes \f$y = \tanh(x)\f$.
 * Backward: \f$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 - y^2)\f$.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Tensor of identical shape with values mapped to \f$(-1, 1)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> tanh_op(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Softmax normalized exponent applied along a specific dimension.
 *
 * @details
 * Computes \f$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}\f$ along
 * axis `dim`.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param dim Axis along which normalization is applied.
 * @return Tensor of identical shape whose slices along `dim` sum to 1.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

}  // namespace scorch

#endif  // !_SCORCH_OPS_ACTIVATION_H_
