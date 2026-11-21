#ifndef _SCORCH_OPS_TRIGONOMETRIC_H_
#define _SCORCH_OPS_TRIGONOMETRIC_H_

/**
 * @file trigonometric.h
 * @brief Differentiable trigonometric and advanced mathematical functions.
 *
 * @details
 * Provides autograd-aware transcendental and mathematical functions:
 * - Circular trigonometric: `sin`, `cos`, `tan`
 * - Inverse trigonometric: `asin`, `acos`, `atan`
 * - Hyperbolic: `sinh`, `cosh`, `tanh`
 * - Logarithms: `loge` (natural), `log2`, `log10`
 * - Powers and roots: `sqrt`, `rsqrt`, `abs`
 *
 * @ingroup ops
 */

#include <memory>

#include "ops/activation.h"
#include "ops/arithmetic.h"
#include "tensor.h"

namespace scorch {

// -----------------------------------------------------------------------------
// Trigonometric functions
// -----------------------------------------------------------------------------

/**
 * @brief Element-wise sine (\f$y = \sin(x)\f$).
 * Backward: \f$\cos(x)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sin(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise cosine (\f$y = \cos(x)\f$).
 * Backward: \f$-\sin(x)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cos(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise tangent (\f$y = \tan(x)\f$).
 * Backward: \f$1 / \cos^2(x)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> tan(const std::shared_ptr<Tensor<T>>& x);

// -----------------------------------------------------------------------------
// Inverse trigonometric functions
// -----------------------------------------------------------------------------

/**
 * @brief Element-wise inverse sine (\f$y = \arcsin(x)\f$).
 * Backward: \f$1 / \sqrt{1 - x^2}\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> asin(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise inverse cosine (\f$y = \arccos(x)\f$).
 * Backward: \f$-1 / \sqrt{1 - x^2}\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> acos(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise inverse tangent (\f$y = \arctan(x)\f$).
 * Backward: \f$1 / (1 + x^2)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> atan(const std::shared_ptr<Tensor<T>>& x);

// -----------------------------------------------------------------------------
// Hyperbolic functions
// -----------------------------------------------------------------------------

/**
 * @brief Element-wise hyperbolic sine (\f$y = \sinh(x)\f$).
 * Backward: \f$\cosh(x)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sinh(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise hyperbolic cosine (\f$y = \cosh(x)\f$).
 * Backward: \f$\sinh(x)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cosh(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise hyperbolic tangent (\f$y = \tanh(x)\f$).
 * Shorthand alias delegating to `tanh_op(x)`.
 */
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> tanh(const std::shared_ptr<Tensor<T>>& x) {
  return tanh_op(x);
}

// -----------------------------------------------------------------------------
// Logarithmic functions
// -----------------------------------------------------------------------------

/**
 * @brief Element-wise natural logarithm (\f$y = \ln(x)\f$).
 * Shorthand alias delegating to `log(x)`.
 */
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> loge(const std::shared_ptr<Tensor<T>>& x) {
  return log(x);
}

/**
 * @brief Element-wise base-2 logarithm (\f$y = \log_2(x)\f$).
 * Backward: \f$1 / (x \ln(2))\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log2(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise base-10 logarithm (\f$y = \log_{10}(x)\f$).
 * Backward: \f$1 / (x \ln(10))\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log10(const std::shared_ptr<Tensor<T>>& x);

// -----------------------------------------------------------------------------
// Roots and absolute value
// -----------------------------------------------------------------------------

/**
 * @brief Element-wise square root (\f$y = \sqrt{x}\f$).
 * Backward: \f$1 / (2\sqrt{x})\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sqrt(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise reciprocal square root (\f$y = 1 / \sqrt{x}\f$).
 * Backward: \f$-1 / (2 x^{3/2})\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> rsqrt(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise absolute value (\f$y = |x|\f$).
 * Backward: \f$\text{sgn}(x)\f$.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> abs(const std::shared_ptr<Tensor<T>>& x);

}  // namespace scorch

#endif  // !_SCORCH_OPS_TRIGONOMETRIC_H_
