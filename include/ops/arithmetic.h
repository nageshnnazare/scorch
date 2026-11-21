#ifndef _SCORCH_OPS_ARITHMETIC_H_
#define _SCORCH_OPS_ARITHMETIC_H_

/**
 * @file arithmetic.h
 * @brief Differentiable element-wise arithmetic and scalar operations.
 *
 * @details
 * Defines autograd-enabled basic arithmetic operations:
 * addition, subtraction, multiplication, division, negation, powers, exponential,
 * logarithm, and scalar offsets/scalings.
 *
 * @ingroup ops
 */

#include <memory>

#include "tensor.h"

namespace scorch {

/**
 * @brief Element-wise addition of two tensors (\f$y = a + b\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param a First operand tensor.
 * @param b Second operand tensor (must have same shape as `a`).
 * @return Sum tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise subtraction of two tensors (\f$y = a - b\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param a First operand tensor.
 * @param b Second operand tensor (must have same shape as `a`).
 * @return Difference tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sub(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise multiplication of two tensors (\f$y = a \odot b\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param a First operand tensor.
 * @param b Second operand tensor (must have same shape as `a`).
 * @return Element-wise product tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise division of two tensors (\f$y = a / b\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param a Numerator tensor.
 * @param b Denominator tensor (must have same shape as `a`).
 * @return Element-wise quotient tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> div(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise unary negation (\f$y = -x\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Negated tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> neg(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise power function (\f$y = x^{\text{exponent}}\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Base tensor.
 * @param exponent Power exponent scalar.
 * @return Result tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> pow(const std::shared_ptr<Tensor<T>>& x, T exponent);

/**
 * @brief Element-wise natural exponential (\f$y = e^x\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @return Exponential tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> exp(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise natural logarithm (\f$y = \ln(x)\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor (values should be positive).
 * @return Logarithm tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise scalar addition (\f$y = x + c\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param scalar Constant value added to each element.
 * @return Result tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> add_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

/**
 * @brief Element-wise scalar multiplication (\f$y = x \cdot c\f$).
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input tensor.
 * @param scalar Constant scaling multiplier.
 * @return Scaled tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

}  // namespace scorch

#endif  // !_SCORCH_OPS_ARITHMETIC_H_
