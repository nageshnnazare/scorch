#ifndef _AG_TENSOR_H_
#define _AG_TENSOR_H_

/**
 * @file tensor.h
 * @brief Multidimensional Tensor, TensorView, and Autograd engine core declarations.
 *
 * =============================================================================
 * MEMORY LAYOUT & SUBSCRIPT ACCESS ILLUSTRATION
 * =============================================================================
 *
 * 1. Physical Storage (Row-Major Contiguous 1D Buffer):
 *
 *    Consider a 2D Tensor with Shape = {2, 3}:
 *
 *    Logical 2D View:
 *    +------------------------------------+
 *    | Row 0:  (0, 0)   (0, 1)   (0, 2)   |
 *    | Row 1:  (1, 0)   (1, 1)   (1, 2)   |
 *    +------------------------------------+
 *
 *    Physical Storage (`std::vector<T> data`):
 *    +---------+---------+---------+---------+---------+---------+
 *    |  (0, 0) |  (0, 1) |  (0, 2) |  (1, 0) |  (1, 1) |  (1, 2) |
 *    +---------+---------+---------+---------+---------+---------+
 *    Flat Idx:     0         1         2         3         4         5
 *              | <------ Row 0 ------> | <------ Row 1 ------> |
 *
 * 2. Multi-dimensional Subscript Traversal (e.g., `t[1][2]`):
 *
 *    Chained `[]` operators narrow dimensions one-by-one using TensorView proxies:
 *
 *    [Tensor t] (Shape {2, 3})
 *        |
 *    t[1]|  Tensor::operator[](1)
 *        |  stride(dim 0) = shape[1] = 3
 *        |  offset = 1 * 3 = 3
 *        v
 *    [TensorView] (dim: 1, offset: 3, shape: {2, 3})
 *        |
 *     [2]|  TensorView::operator[](2)
 *        |  stride(dim 1) = 1
 *        |  offset = 3 + 2 * 1 = 5
 *        v
 *    [TensorView] (dim: 2, offset: 5, shape: {2, 3})
 *        |
 *        |  operator T&() (Implicit conversion to scalar reference)
 *        v
 *    base[5] -> (1, 2)
 *
 * 3. General N-Dimensional Stride Formula:
 *
 *    For Shape {S_0, S_1, ..., S_{n-1}}:
 *      stride(d) = S_{d+1} * S_{d+2} * ... * S_{n-1}   (stride(n-1) = 1)
 *      flat_index = sum_{d=0}^{n-1} (coord_d * stride(d))
 *
 * =============================================================================
 */

#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace AG {

/**
 * @concept TensorScalar
 * @brief C++20 concept restricting tensor element types to supported types: float, double, or int.
 */
template <typename T>
concept TensorScalar =
    std::same_as<std::remove_cv_t<T>, float> || std::same_as<std::remove_cv_t<T>, double> ||
    std::same_as<std::remove_cv_t<T>, int>;

/**
 * @brief Represents the dimensional extents of a Tensor (e.g., {batch, channels, height, width}).
 */
using Shape = std::vector<std::size_t>;

/**
 * @brief Lightweight proxy view for chained multidimensional subscript indexing (e.g.,
 * `(*t)[i][j]`).
 *
 * Each bracket operation steps forward through the dimension hierarchy, tracking
 * cumulative linear memory offset until the final scalar element is reached.
 *
 * @tparam T Element data type (must satisfy TensorScalar).
 */
template <TensorScalar T>
struct TensorView {
  T* base;             ///< Pointer to underlying flat storage buffer.
  const Shape* shape;  ///< Non-owning pointer to the tensor shape.
  int dim;             ///< Current dimension index being resolved.
  std::size_t offset;  ///< Current flat linear offset into base buffer.

  /**
   * @brief Calculates the linear memory stride of the current dimension.
   * @return Number of elements to step per unit in this dimension.
   */
  std::size_t stride() const {
    std::size_t s = 1;
    for (int d = dim + 1; d < static_cast<int>(shape->size()); ++d) s *= (*shape)[d];
    return s;
  }

  /**
   * @brief Advances one dimension deeper in the tensor view.
   * @param i Index along the current dimension.
   * @return A new TensorView narrowed to the next dimension.
   */
  TensorView operator[](std::size_t i) { return {base, shape, dim + 1, offset + i * stride()}; }

  /**
   * @brief Implicit conversion to scalar reference for read/write access.
   */
  operator T&() { return base[offset]; }

  /**
   * @brief Implicit conversion to const scalar reference for read access.
   */
  operator const T&() const { return base[offset]; }

  /**
   * @brief Assigns a scalar value to the element at the current linear offset.
   * @param val Scalar value to store.
   * @return Reference to this TensorView.
   */
  TensorView& operator=(const T& val) {
    base[offset] = val;
    return *this;
  }
};

/**
 * @brief Computes the total number of scalar elements represented by a Shape.
 * @param shape Tensor dimensions.
 * @return Total element count (product of all dimensions, or 1 for scalar 0D shape).
 */
std::size_t numel(const Shape& shape);

/**
 * @brief Converts multi-dimensional coordinates into a row-major linear flat index.
 * @param shape Shape of the tensor.
 * @param coords Multi-dimensional coordinate vector matching shape rank.
 * @return Flat linear memory index.
 */
std::size_t flatid(const Shape& shape, const std::vector<std::size_t>& coords);

/**
 * @brief Returns the pointer to the Tensor currently being backpropagated.
 * Used internally by autograd backward closures to read outgoing gradients.
 */
void* get_current_backward_tensor();

// Forward declarations of differentiable tensor operations returning shared_ptr
template <TensorScalar T>
struct Tensor;

// --- Arithmetic ---
template <TensorScalar T>
std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sub(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> div(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> neg(const std::shared_ptr<Tensor<T>>& x);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> pow(const std::shared_ptr<Tensor<T>>& x, T exponent);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> exp(const std::shared_ptr<Tensor<T>>& x);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& x);

// --- Scalar-Tensor ---
template <TensorScalar T>
std::shared_ptr<Tensor<T>> add_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar);

// --- Matrix ---
template <TensorScalar T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a,
                                  const std::shared_ptr<Tensor<T>>& b);

// --- Activations ---
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

// --- Shape / View ---
template <TensorScalar T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& x);
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
std::shared_ptr<Tensor<T>> cat(std::initializer_list<std::shared_ptr<Tensor<T>>> tensors,
                               std::size_t dim);

// --- Reductions ---
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reduce(const std::shared_ptr<Tensor<T>>& x);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>>& x);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> max(const std::shared_ptr<Tensor<T>>& x);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sum_dim(const std::shared_ptr<Tensor<T>>& x, std::size_t dim);

// --- Loss Functions ---
template <TensorScalar T>
std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& pred,
                                    const std::shared_ptr<Tensor<T>>& target);
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& logits,
                                              const std::shared_ptr<Tensor<int>>& targets);

/**
 * @brief N-dimensional dense array container with autograd backpropagation support.
 *
 * Inherits from std::enable_shared_from_this to allow shared ownership across the DAG.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar: float, double, or int).
 */
template <TensorScalar T>
struct Tensor : public std::enable_shared_from_this<Tensor<T>> {
  Shape shape;          ///< Dimensions of the tensor.
  std::vector<T> data;  ///< Contiguous row-major storage for tensor values.
  std::string op_name;  ///< Name of operation that created this tensor (for graph visualization).

  bool requires_grad = false;  ///< Flag indicating if gradients should be tracked.
  std::vector<float> grad;     ///< Accumulated gradients with respect to this tensor.
  std::vector<std::shared_ptr<Tensor>> parents;  ///< Predecessor nodes in the computation graph.
  std::function<void()> backward_fn;             ///< Backward closure for gradient accumulation.

  // ---------------------------------------------------------------------------
  // Friend Declarations for Differentiable Operations
  // ---------------------------------------------------------------------------

  // Arithmetic
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> add(const std::shared_ptr<Tensor<U>>& a,
                                        const std::shared_ptr<Tensor<U>>& b);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> sub(const std::shared_ptr<Tensor<U>>& a,
                                        const std::shared_ptr<Tensor<U>>& b);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> mul(const std::shared_ptr<Tensor<U>>& a,
                                        const std::shared_ptr<Tensor<U>>& b);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> div(const std::shared_ptr<Tensor<U>>& a,
                                        const std::shared_ptr<Tensor<U>>& b);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> neg(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> pow(const std::shared_ptr<Tensor<U>>& x, U exponent);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> exp(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> log(const std::shared_ptr<Tensor<U>>& x);

  // Scalar-Tensor
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> add_scalar(const std::shared_ptr<Tensor<U>>& x, U scalar);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> mul_scalar(const std::shared_ptr<Tensor<U>>& x, U scalar);

  // Matrix
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> matmul(const std::shared_ptr<Tensor<U>>& a,
                                           const std::shared_ptr<Tensor<U>>& b);

  // Activations
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> relu(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> leaky_relu(const std::shared_ptr<Tensor<U>>& x, U alpha);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> sigmoid(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> tanh_op(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> softmax(const std::shared_ptr<Tensor<U>>& x, std::size_t dim);

  // Shape / View
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> transpose(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> reshape(const std::shared_ptr<Tensor<U>>& x, Shape new_shape);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> flatten(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> squeeze(const std::shared_ptr<Tensor<U>>& x, std::size_t dim);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> unsqueeze(const std::shared_ptr<Tensor<U>>& x, std::size_t dim);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> cat(const std::vector<std::shared_ptr<Tensor<U>>>& tensors,
                                        std::size_t dim);

  // Reductions
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> reduce(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> mean(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> max(const std::shared_ptr<Tensor<U>>& x);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> sum_dim(const std::shared_ptr<Tensor<U>>& x, std::size_t dim);

  // Loss Functions
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> mse_loss(const std::shared_ptr<Tensor<U>>& pred,
                                             const std::shared_ptr<Tensor<U>>& target);
  template <TensorScalar U>
  friend std::shared_ptr<Tensor<U>> cross_entropy_loss(const std::shared_ptr<Tensor<U>>& logits,
                                                       const std::shared_ptr<Tensor<int>>& targets);

  // ---------------------------------------------------------------------------
  // Member Operator Overloads (dispatched via shared_from_this)
  // ---------------------------------------------------------------------------

  std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other) {
    return add(this->shared_from_this(), other);
  }

  std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& other) {
    return sub(this->shared_from_this(), other);
  }

  std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& other) {
    return mul(this->shared_from_this(), other);
  }

  std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& other) {
    return div(this->shared_from_this(), other);
  }

  std::shared_ptr<Tensor> operator-() { return neg(this->shared_from_this()); }

  std::shared_ptr<Tensor> operator+(T scalar) {
    return add_scalar(this->shared_from_this(), scalar);
  }

  std::shared_ptr<Tensor> operator*(T scalar) {
    return mul_scalar(this->shared_from_this(), scalar);
  }

  // ---------------------------------------------------------------------------
  // Subscript Operators
  // ---------------------------------------------------------------------------

  /**
   * @brief Multi-dimensional indexing operator (mutable).
   * @param i Index along the outermost dimension.
   * @return A TensorView resolving index `i`.
   */
  TensorView<T> operator[](std::size_t i);

  /**
   * @brief Multi-dimensional indexing operator (const).
   * @param i Index along the outermost dimension.
   * @return A const TensorView resolving index `i`.
   */
  const TensorView<const T> operator[](std::size_t i) const;

  // ---------------------------------------------------------------------------
  // Factory Methods (Returning std::shared_ptr<Tensor<T>>)
  // ---------------------------------------------------------------------------

  static std::shared_ptr<Tensor<T>> full(Shape shape, T fill_value, bool requires_grad = false);
  static std::shared_ptr<Tensor<T>> zeros(Shape shape, bool requires_grad = false);
  static std::shared_ptr<Tensor<T>> ones(Shape shape, bool requires_grad = false);
  static std::shared_ptr<Tensor<T>> rand(Shape shape, bool requires_grad = false);
  static std::shared_ptr<Tensor<T>> from(Shape shape, std::vector<T> data,
                                         bool requires_grad = false);

  // ---------------------------------------------------------------------------
  // Autograd
  // ---------------------------------------------------------------------------

  void zero_grad();
  void backward();

  // ---------------------------------------------------------------------------
  // Formatting
  // ---------------------------------------------------------------------------

  void print(const char* name = "") const;

 private:
  void print_dim(int dim, std::size_t offset, const char* base_indent) const;
};

// -----------------------------------------------------------------------------
// Free Operator Overloads on std::shared_ptr<Tensor<T>>
// -----------------------------------------------------------------------------

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& a,
                                            const std::shared_ptr<Tensor<T>>& b) {
  return add(a, b);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& a,
                                            const std::shared_ptr<Tensor<T>>& b) {
  return sub(a, b);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& a,
                                            const std::shared_ptr<Tensor<T>>& b) {
  return mul(a, b);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator/(const std::shared_ptr<Tensor<T>>& a,
                                            const std::shared_ptr<Tensor<T>>& b) {
  return div(a, b);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& a) {
  return neg(a);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& a, T scalar) {
  return add_scalar(a, scalar);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator+(T scalar, const std::shared_ptr<Tensor<T>>& a) {
  return add_scalar(a, scalar);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& a, T scalar) {
  return mul_scalar(a, scalar);
}

template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> operator*(T scalar, const std::shared_ptr<Tensor<T>>& a) {
  return mul_scalar(a, scalar);
}

}  // namespace AG

#endif  // !_AG_TENSOR_H_
