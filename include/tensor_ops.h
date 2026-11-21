#ifndef _AG_TENSOR_OPS_H_
#define _AG_TENSOR_OPS_H_

/**
 * @file tensor_ops.h
 * @brief Differentiable arithmetic, matrix, activation, reduction, shape, and loss
 *        operations for AG::Tensor returning std::shared_ptr<Tensor<T>>.
 *
 * Implements forward computation and automatic reverse-mode gradient closures.
 * Type-constrained by C++20 concept AG::TensorScalar (float, double, int).
 */

#include <initializer_list>
#include <memory>
#include <vector>

#include "tensor.h"

namespace AG {

// =============================================================================
// Element-wise Arithmetic Operations
// =============================================================================

/**
 * @brief Element-wise addition: out = a + b.
 *
 * Gradient:  dL/da += dL/dout,  dL/db += dL/dout
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise subtraction: out = a - b.
 *
 * Gradient:  dL/da += dL/dout,  dL/db -= dL/dout
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sub(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise multiplication (Hadamard product): out = a * b.
 *
 * Gradient:  dL/da += dL/dout * b,  dL/db += dL/dout * a
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise division: out = a / b.
 *
 * Gradient:  dL/da += dL/dout / b,  dL/db += -dL/dout * a / b²
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> div(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);

/**
 * @brief Element-wise negation: out = -x.
 *
 * Gradient:  dL/dx += -dL/dout
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> neg(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise power: out[i] = x[i]^exponent.
 *
 * Gradient:  dL/dx += dL/dout * exponent * x^(exponent-1)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> pow(const std::shared_ptr<Tensor<T>>& x, T exponent);

/**
 * @brief Element-wise exponential: out = exp(x).
 *
 * Gradient:  dL/dx += dL/dout * exp(x)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> exp(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise natural logarithm: out = ln(x).
 *
 * Gradient:  dL/dx += dL/dout / x
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& x);

// =============================================================================
// Scalar-Tensor Operations
// =============================================================================

/**
 * @brief Add a scalar to every element: out[i] = x[i] + scalar.
 *
 * Gradient:  dL/dx += dL/dout  (scalar has no gradient)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> add_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

/**
 * @brief Multiply every element by a scalar: out[i] = x[i] * scalar.
 *
 * Gradient:  dL/dx += dL/dout * scalar
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

// =============================================================================
// Matrix Operations
// =============================================================================

/**
 * @brief 2D Matrix multiplication: out = a @ b.
 *
 * For A (M×K) and B (K×N), computes C (M×N).
 *
 * Gradient:  dL/da += dL/dout @ b^T,  dL/db += a^T @ dL/dout
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a,
                                  const std::shared_ptr<Tensor<T>>& b);

// =============================================================================
// Activation Functions
// =============================================================================

/**
 * @brief Element-wise Rectified Linear Unit: out = max(0, x).
 *
 * Gradient:  dL/dx += (x > 0) ? dL/dout : 0
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise Leaky ReLU: out = (x > 0) ? x : alpha * x.
 *
 * Gradient:  dL/dx += (x > 0) ? dL/dout : alpha * dL/dout
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> leaky_relu(const std::shared_ptr<Tensor<T>>& x, T alpha);

/**
 * @brief Element-wise sigmoid activation: out = 1 / (1 + exp(-x)).
 *
 * Gradient:  dL/dx += dL/dout * σ(x) * (1 - σ(x))
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Element-wise hyperbolic tangent activation: out = tanh(x).
 *
 * Gradient:  dL/dx += dL/dout * (1 - tanh²(x))
 *
 * Named tanh_op to avoid conflict with std::tanh.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> tanh_op(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Softmax along a given dimension of a rank-2 tensor.
 *
 * out[i][j] = exp(x[i][j]) / sum_k(exp(x[i][k]))   (for dim=1)
 *
 * Gradient:  dL/dx_i += s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

// =============================================================================
// Shape / View Operations
// =============================================================================

/**
 * @brief Transposes a rank-2 matrix: out = x^T.
 *
 * Gradient:  dL/dx += (dL/dout)^T
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Reshape tensor to new_shape (must have same numel).
 *
 * Gradient:  dL/dx += reshape(dL/dout, original_shape)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reshape(const std::shared_ptr<Tensor<T>>& x, Shape new_shape);

/**
 * @brief Flatten tensor to 1D: out.shape = {numel(x)}.
 *
 * Gradient:  dL/dx += reshape(dL/dout, original_shape)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> flatten(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Remove a dimension of size 1 at position dim.
 *
 * Gradient:  dL/dx += unsqueeze(dL/dout, dim)
 *
 * @throws std::invalid_argument if shape[dim] != 1.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> squeeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

/**
 * @brief Insert a dimension of size 1 at position dim.
 *
 * Gradient:  dL/dx += squeeze(dL/dout, dim)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> unsqueeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

/**
 * @brief Concatenate tensors along a given dimension.
 *
 * All tensors must have the same shape except along the concatenation dimension.
 *
 * Gradient:  dL/dx_k += slice of dL/dout corresponding to tensor k
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cat(const std::vector<std::shared_ptr<Tensor<T>>>& tensors,
                               std::size_t dim);

/**
 * @brief Concatenate tensors provided via braced initializer list of shared_ptr.
 */
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> cat(std::initializer_list<std::shared_ptr<Tensor<T>>> tensors,
                                      std::size_t dim) {
  return cat(std::vector<std::shared_ptr<Tensor<T>>>(tensors.begin(), tensors.end()), dim);
}

// =============================================================================
// Reduction Operations
// =============================================================================

/**
 * @brief Global sum reduction: out = sum(x), shape {1}.
 *
 * Gradient:  dL/dx_i += dL/dout  (broadcast)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reduce(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Global mean reduction: out = mean(x), shape {1}.
 *
 * Gradient:  dL/dx_i += dL/dout / numel(x)
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Global max reduction: out = max(x), shape {1}.
 *
 * Gradient:  dL/dx_i += (x_i == max) ? dL/dout : 0
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> max(const std::shared_ptr<Tensor<T>>& x);

/**
 * @brief Sum reduction along a specific dimension (rank-2 tensors).
 *
 * dim=0: out shape {1, cols}, sum down rows.
 * dim=1: out shape {rows, 1}, sum across columns.
 *
 * Gradient:  dL/dx += broadcast dL/dout along reduced dimension.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sum_dim(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

// =============================================================================
// Loss Functions
// =============================================================================

/**
 * @brief Mean Squared Error loss: out = mean((pred - target)²), shape {1}.
 *
 * Gradient:  dL/dpred += dL/dout * 2 * (pred - target) / numel
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& pred,
                                    const std::shared_ptr<Tensor<T>>& target);

/**
 * @brief Cross-Entropy loss with integer class targets.
 *
 * logits: shape {N, C}  (batch × classes)
 * targets: shape {N}    (class indices, Tensor<int>)
 *
 * loss = mean_i( -logits[i][t_i] + log(sum_j(exp(logits[i][j]))) )
 *
 * Gradient:  dL/dlogits[i][j] += (softmax[i][j] - 1_{j==t_i}) / N
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& logits,
                                              const std::shared_ptr<Tensor<int>>& targets);

}  // namespace AG

#endif  // !_AG_TENSOR_OPS_H_
