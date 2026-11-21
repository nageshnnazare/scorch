#ifndef _SCORCH_OPS_MATRIX_H_
#define _SCORCH_OPS_MATRIX_H_

/**
 * @file matrix.h
 * @brief Differentiable 2D matrix multiplication and transposition.
 *
 * @details
 * Provides autograd-aware 2-D matrix operations:
 * - `matmul(a, b)`: Matrix product \f$C = A B\f$ with gradient propagation \f$\frac{\partial
 * L}{\partial A} = \frac{\partial L}{\partial C} B^T\f$ and \f$\frac{\partial L}{\partial B} = A^T
 * \frac{\partial L}{\partial C}\f$.
 * - `transpose(x)`: Matrix transpose \f$Y = X^T\f$ with gradient propagation \f$\frac{\partial
 * L}{\partial X} = (\frac{\partial L}{\partial Y})^T\f$.
 *
 * @ingroup ops
 */

#include <memory>

#include "tensor.h"

namespace scorch {

/**
 * @brief Performs 2D matrix multiplication of two tensors.
 *
 * @details
 * Evaluates \f$C = A \times B\f$ where \f$A\f$ has shape \f$\{M, K\}\f$ and \f$B\f$ has shape
 * \f$\{K, N\}\f$. The resulting tensor has shape \f$\{M, N\}\f$.
 *
 * If either input requires gradient tracking, backward gradient closures are attached:
 * \f[
 *   \nabla_A = \nabla_C \times B^T, \quad \nabla_B = A^T \times \nabla_C
 * \f]
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param a Left operand matrix of shape \f$\{M, K\}\f$.
 * @param b Right operand matrix of shape \f$\{K, N\}\f$.
 * @return Shared pointer to resulting matrix of shape \f$\{M, N\}\f$.
 * @throws std::invalid_argument If inputs are not 2D or inner dimensions do not match.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a,
                                  const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Performs 2D matrix transposition.
 *
 * @details
 * Swaps rows and columns for a 2D tensor of shape \f$\{M, N\}\f$, returning a tensor
 * of shape \f$\{N, M\}\f$.  Autograd backward propagates the transposed gradient.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param x Input 2D tensor of shape \f$\{M, N\}\f$.
 * @return Shared pointer to transposed tensor of shape \f$\{N, M\}\f$.
 * @throws std::invalid_argument If input is not a 2D tensor.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& x);

}  // namespace scorch

#endif  // !_SCORCH_OPS_MATRIX_H_
