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
 * @brief Lightweight proxy view for chained multidimensional subscript indexing (e.g., `t[i][j]`).
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

// Forward declarations of differentiable tensor operations constrained by TensorScalar
template <TensorScalar T>
struct Tensor;

// --- Arithmetic ---
template <TensorScalar T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);
template <TensorScalar T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b);
template <TensorScalar T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b);
template <TensorScalar T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b);
template <TensorScalar T>
Tensor<T> neg(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> pow(const Tensor<T>& x, T exponent);
template <TensorScalar T>
Tensor<T> exp(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> log(const Tensor<T>& x);

// --- Scalar-Tensor ---
template <TensorScalar T>
Tensor<T> add_scalar(const Tensor<T>& x, T scalar);
template <TensorScalar T>
Tensor<T> mul_scalar(const Tensor<T>& x, T scalar);

// --- Matrix ---
template <TensorScalar T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

// --- Activations ---
template <TensorScalar T>
Tensor<T> relu(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> leaky_relu(const Tensor<T>& x, T alpha);
template <TensorScalar T>
Tensor<T> sigmoid(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> tanh_op(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> softmax(const Tensor<T>& x, std::size_t dim);

// --- Shape / View ---
template <TensorScalar T>
Tensor<T> transpose(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> reshape(const Tensor<T>& x, Shape new_shape);
template <TensorScalar T>
Tensor<T> flatten(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> squeeze(const Tensor<T>& x, std::size_t dim);
template <TensorScalar T>
Tensor<T> unsqueeze(const Tensor<T>& x, std::size_t dim);
template <TensorScalar T>
Tensor<T> cat(const std::vector<const Tensor<T>*>& tensors, std::size_t dim);
template <TensorScalar T>
Tensor<T> cat(const std::vector<Tensor<T>>& tensors, std::size_t dim);
template <TensorScalar T>
Tensor<T> cat(std::initializer_list<const Tensor<T>*> tensors, std::size_t dim);
template <TensorScalar T>
Tensor<T> cat(std::initializer_list<Tensor<T>*> tensors, std::size_t dim);

// --- Reductions ---
template <TensorScalar T>
Tensor<T> reduce(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> mean(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> max(const Tensor<T>& x);
template <TensorScalar T>
Tensor<T> sum_dim(const Tensor<T>& x, std::size_t dim);

// --- Loss Functions ---
template <TensorScalar T>
Tensor<T> mse_loss(const Tensor<T>& pred, const Tensor<T>& target);
template <TensorScalar T>
Tensor<T> cross_entropy_loss(const Tensor<T>& logits, const Tensor<int>& targets);

/**
 * @brief N-dimensional dense array container with autograd backpropagation support.
 *
 * Supports multi-dimensional subscripting, initialization factories, gradient storage,
 * and computation graph dependency recording for automatic differentiation.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar: float, double, or int).
 */
template <TensorScalar T>
struct Tensor {
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
  friend Tensor<U> add(const Tensor<U>& a, const Tensor<U>& b);
  template <TensorScalar U>
  friend Tensor<U> sub(const Tensor<U>& a, const Tensor<U>& b);
  template <TensorScalar U>
  friend Tensor<U> mul(const Tensor<U>& a, const Tensor<U>& b);
  template <TensorScalar U>
  friend Tensor<U> div(const Tensor<U>& a, const Tensor<U>& b);
  template <TensorScalar U>
  friend Tensor<U> neg(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> pow(const Tensor<U>& x, U exponent);
  template <TensorScalar U>
  friend Tensor<U> exp(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> log(const Tensor<U>& x);

  // Scalar-Tensor
  template <TensorScalar U>
  friend Tensor<U> add_scalar(const Tensor<U>& x, U scalar);
  template <TensorScalar U>
  friend Tensor<U> mul_scalar(const Tensor<U>& x, U scalar);

  // Matrix
  template <TensorScalar U>
  friend Tensor<U> matmul(const Tensor<U>& a, const Tensor<U>& b);

  // Activations
  template <TensorScalar U>
  friend Tensor<U> relu(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> leaky_relu(const Tensor<U>& x, U alpha);
  template <TensorScalar U>
  friend Tensor<U> sigmoid(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> tanh_op(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> softmax(const Tensor<U>& x, std::size_t dim);

  // Shape / View
  template <TensorScalar U>
  friend Tensor<U> transpose(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> reshape(const Tensor<U>& x, Shape new_shape);
  template <TensorScalar U>
  friend Tensor<U> flatten(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> squeeze(const Tensor<U>& x, std::size_t dim);
  template <TensorScalar U>
  friend Tensor<U> unsqueeze(const Tensor<U>& x, std::size_t dim);
  template <TensorScalar U>
  friend Tensor<U> cat(const std::vector<const Tensor<U>*>& tensors, std::size_t dim);
  template <TensorScalar U>
  friend Tensor<U> cat(const std::vector<Tensor<U>>& tensors, std::size_t dim);

  // Reductions
  template <TensorScalar U>
  friend Tensor<U> reduce(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> mean(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> max(const Tensor<U>& x);
  template <TensorScalar U>
  friend Tensor<U> sum_dim(const Tensor<U>& x, std::size_t dim);

  // Loss Functions
  template <TensorScalar U>
  friend Tensor<U> mse_loss(const Tensor<U>& pred, const Tensor<U>& target);
  template <TensorScalar U>
  friend Tensor<U> cross_entropy_loss(const Tensor<U>& logits, const Tensor<int>& targets);

  // ---------------------------------------------------------------------------
  // Operator Overloads
  // ---------------------------------------------------------------------------

  /** @brief Element-wise addition: out = a + b. */
  Tensor operator+(const Tensor& other) const { return add(*this, other); }

  /** @brief Element-wise subtraction: out = a - b. */
  Tensor operator-(const Tensor& other) const { return sub(*this, other); }

  /** @brief Element-wise multiplication: out = a * b. */
  Tensor operator*(const Tensor& other) const { return mul(*this, other); }

  /** @brief Element-wise division: out = a / b. */
  Tensor operator/(const Tensor& other) const { return div(*this, other); }

  /** @brief Unary negation: out = -x. */
  Tensor operator-() const { return neg(*this); }

  /** @brief Add scalar to all elements: out = x + s. */
  Tensor operator+(T scalar) const { return add_scalar(*this, scalar); }

  /** @brief Multiply all elements by scalar: out = x * s. */
  Tensor operator*(T scalar) const { return mul_scalar(*this, scalar); }

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
  // Factory Methods
  // ---------------------------------------------------------------------------

  static Tensor<T> full(Shape shape, T fill_value, bool requires_grad = false);
  static Tensor<T> zeros(Shape shape, bool requires_grad = false);
  static Tensor<T> ones(Shape shape, bool requires_grad = false);
  static Tensor<T> rand(Shape shape, bool requires_grad = false);
  static Tensor<T> from(Shape shape, std::vector<T> data, bool requires_grad = false);

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

}  // namespace AG

#endif  // !_AG_TENSOR_H_
