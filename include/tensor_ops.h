#ifndef _AG_TENSOR_OPS_H_
#define _AG_TENSOR_OPS_H_

/**
 * @file tensor_ops.h
 * @brief Differentiable arithmetic, matrix, and reduction operations for AG::Tensor.
 *
 * Implements forward computation and automatic reverse-mode gradient closures for
 * element-wise addition, element-wise multiplication, matrix multiplication,
 * rectified linear unit (ReLU), 2D transposition, and global summation reduction.
 * Type-constrained by C++20 concept AG::TensorScalar (float, double, int).
 */

#include "tensor.h"

namespace AG {

/**
 * @brief Element-wise addition of two identically-shaped tensors: out = a + b.
 *
 * Gradient:
 *   dL/da += dL/dout
 *   dL/db += dL/dout
 *
 * @tparam T Underlying scalar type (must satisfy AG::TensorScalar).
 * @param a First operand tensor.
 * @param b Second operand tensor.
 * @return Result tensor with shape equal to operand shape.
 * @throws std::invalid_argument if a.shape != b.shape.
 */
template <TensorScalar T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);

/**
 * @brief Element-wise multiplication (Hadamard product) of two tensors: out = a * b.
 *
 * Gradient:
 *   dL/da += dL/dout * b
 *   dL/db += dL/dout * a
 *
 * @tparam T Underlying scalar type (must satisfy AG::TensorScalar).
 * @param a First operand tensor.
 * @param b Second operand tensor.
 * @return Result tensor with shape equal to operand shape.
 * @throws std::invalid_argument if a.shape != b.shape.
 */
template <TensorScalar T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b);

/**
 * @brief 2D Matrix multiplication: out = a @ b.
 *
 * For input matrices A (M x K) and B (K x N), computes C (M x N).
 *
 * Gradient:
 *   dL/da += dL/dout @ b^T
 *   dL/db += a^T @ dL/dout
 *
 * @tparam T Underlying scalar type (must satisfy AG::TensorScalar).
 * @param a Left operand matrix of shape {M, K}.
 * @param b Right operand matrix of shape {K, N}.
 * @return Result matrix of shape {M, N}.
 * @throws std::invalid_argument if either operand is not rank-2 or if inner dimensions mismatch.
 */
template <TensorScalar T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

/**
 * @brief Element-wise Rectified Linear Unit activation: out = max(0, x).
 *
 * Gradient:
 *   dL/dx += (x > 0) ? dL/dout : 0
 *
 * @tparam T Underlying scalar type (must satisfy AG::TensorScalar).
 * @param x Input tensor.
 * @return Result tensor with the same shape as x.
 */
template <TensorScalar T>
Tensor<T> relu(const Tensor<T>& x);

/**
 * @brief Transposes a rank-2 matrix: out = x^T.
 *
 * Swaps dimensions of matrix X (M x N) into shape {N, M}.
 *
 * Gradient:
 *   dL/dx += (dL/dout)^T
 *
 * @tparam T Underlying scalar type (must satisfy AG::TensorScalar).
 * @param x Input rank-2 tensor of shape {M, N}.
 * @return Transposed tensor of shape {N, M}.
 * @throws std::invalid_argument if x is not rank-2.
 */
template <TensorScalar T>
Tensor<T> transpose(const Tensor<T>& x);

/**
 * @brief Global sum reduction over all elements: out = sum(x).
 *
 * Computes a scalar tensor of shape {1} containing the sum of all elements in x.
 *
 * Gradient:
 *   dL/dx_i += dL/dout (broadcast to every element of x)
 *
 * @tparam T Underlying scalar type (must satisfy AG::TensorScalar).
 * @param x Input tensor of any rank.
 * @return Scalar tensor of shape {1}.
 */
template <TensorScalar T>
Tensor<T> reduce(const Tensor<T>& x);

}  // namespace AG

#endif  // !_AG_TENSOR_OPS_H_
