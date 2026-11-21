/**
 * @file test_tensor.cc
 * @brief Zero-dependency unit test suite and runner for AG::Tensor and AG::tensor_ops.
 *
 * Provides a minimal lightweight test harness that registers named test cases,
 * executes them, and reports per-test PASS/FAIL status. Can run all tests or
 * target a specific test by name when invoked from CTest.
 */

#include <tensor.h>
#include <tensor_ops.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

/**
 * @brief Represents an individual test case with an identifier and runner callable.
 */
struct TestCase {
  const char* name;
  std::function<void()> fn;
};

/// Global registry of registered test cases.
static std::vector<TestCase> g_tests;

/**
 * @def TEST(name)
 * @brief Registers a test case with the global test harness.
 */
#define TEST(name)                           \
  static void test_##name();                 \
  static bool reg_##name = [] {              \
    g_tests.push_back({#name, test_##name}); \
    return true;                             \
  }();                                       \
  static void test_##name()

/**
 * @def EXPECT(cond)
 * @brief Asserts that a condition is true, throwing an exception on failure.
 */
#define EXPECT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      throw std::runtime_error("assertion failed");                           \
    }                                                                         \
  } while (0)

/**
 * @def EXPECT_NEAR(a, b, tol)
 * @brief Asserts that two floating-point values are within an absolute tolerance.
 */
#define EXPECT_NEAR(a, b, tol) EXPECT(std::fabs((a) - (b)) <= (tol))

// ===========================================================================
// Core Tensor Tests
// ===========================================================================

/// Verify shape and total element allocation of zeros factory.
TEST(zeros_shape) {
  auto t = AG::Tensor<float>::zeros({3, 5});
  EXPECT(t.shape.size() == 2);
  EXPECT(t.shape[0] == 3);
  EXPECT(t.shape[1] == 5);
  EXPECT(t.data.size() == 15);
}

/// Verify that zeros tensor elements are initialized to 0.
TEST(zeros_values) {
  auto t = AG::Tensor<float>::zeros({2, 4});
  for (auto v : t.data) EXPECT_NEAR(v, 0.f, 1e-9f);
}

/// Verify that ones tensor elements are initialized to 1.
TEST(ones_values) {
  auto t = AG::Tensor<float>::ones({2, 3});
  for (auto v : t.data) EXPECT_NEAR(v, 1.f, 1e-9f);
}

/// Verify that full tensor elements match the requested fill value.
TEST(full_values) {
  auto t = AG::Tensor<float>::full({3, 3}, 7.f);
  for (auto v : t.data) EXPECT_NEAR(v, 7.f, 1e-9f);
}

/// Verify that rand generates elements in the range [0.0, 1.0).
TEST(rand_range) {
  auto t = AG::Tensor<float>::rand({10, 10});
  for (auto v : t.data) {
    EXPECT(v >= 0.f);
    EXPECT(v < 1.f);
  }
}

/// Verify valid initialization from existing data buffer.
TEST(from_happy) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT(t.shape[0] == 2 && t.shape[1] == 3);
  EXPECT_NEAR(t.data[0], 1.f, 1e-9f);
  EXPECT_NEAR(t.data[5], 6.f, 1e-9f);
}

/// Verify that mismatched data size and shape numel throws std::invalid_argument.
TEST(from_mismatch_throws) {
  bool threw = false;
  try {
    AG::Tensor<float>::from({2, 3}, {1, 2, 3});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

/// Verify multi-dimensional subscript read access.
TEST(indexing_read) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_NEAR(static_cast<float>(t[0][0]), 1.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[0][2]), 3.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[1][0]), 4.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[1][2]), 6.f, 1e-9f);
}

/// Verify multi-dimensional subscript write access and isolation.
TEST(indexing_write) {
  auto t = AG::Tensor<float>::zeros({2, 3});
  t[1][2] = 42.f;
  EXPECT_NEAR(static_cast<float>(t[1][2]), 42.f, 1e-9f);
  // neighbouring elements must be untouched
  EXPECT_NEAR(static_cast<float>(t[1][1]), 0.f, 1e-9f);
}

/// Verify requires_grad initialization flag.
TEST(requires_grad_flag) {
  auto a = AG::Tensor<float>::zeros({2}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::zeros({2}, /*requires_grad=*/false);
  EXPECT(a.requires_grad == true);
  EXPECT(b.requires_grad == false);
}

/// Verify numel calculation and row-major flat coordinate indexing.
TEST(size_and_elements) {
  AG::Shape shape{2, 3};
  auto t = AG::Tensor<float>::from(shape, {1, 2, 3, 4, 5, 6});
  EXPECT(AG::numel(shape) == 6);
  EXPECT(AG::flatid(shape, {1, 1}) == 4);
  EXPECT(t.data[AG::flatid(shape, {1, 2})] == 6);
}

/// Verify gradient initialization and backward pass execution on leaf tensor.
TEST(backward_leaf) {
  auto t = AG::Tensor<float>::zeros({2, 3}, /*requires_grad=*/true);
  t.backward();
  EXPECT(t.grad.size() == 6);
  for (auto g : t.grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

// ===========================================================================
// Differentiable Operator & Autograd Gradient Tests (original ops)
// ===========================================================================

/// Verify forward addition and backpropagated gradients for c = a + b.
TEST(grad_add) {
  auto a = AG::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/true);
  auto c = AG::add(a, b);
  EXPECT_NEAR(c.data[0], 6.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 8.f, 1e-9f);

  c.backward();
  EXPECT_NEAR(a.grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 1.f, 1e-9f);
  EXPECT_NEAR(b.grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(b.grad[1], 1.f, 1e-9f);
}

/// Verify forward multiplication and product rule gradients for c = a * b.
TEST(grad_mul) {
  auto a = AG::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/true);
  auto c = AG::mul(a, b);
  EXPECT_NEAR(c.data[0], 8.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 15.f, 1e-9f);

  c.backward();
  EXPECT_NEAR(a.grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 5.f, 1e-9f);
  EXPECT_NEAR(b.grad[0], 2.f, 1e-9f);
  EXPECT_NEAR(b.grad[1], 3.f, 1e-9f);
}

/// Verify matrix multiplication forward and matrix calculus gradients C = A @ B.
TEST(grad_matmul) {
  auto a = AG::Tensor<float>::from({2, 2}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2, 2}, {5.f, 6.f, 7.f, 8.f}, /*requires_grad=*/true);
  auto c = AG::matmul(a, b);
  // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
  // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
  EXPECT_NEAR(c.data[0], 19.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 22.f, 1e-9f);
  EXPECT_NEAR(c.data[2], 43.f, 1e-9f);
  EXPECT_NEAR(c.data[3], 50.f, 1e-9f);

  c.backward();
  // dA = dC @ B^T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
  EXPECT_NEAR(a.grad[0], 11.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 15.f, 1e-9f);
  EXPECT_NEAR(a.grad[2], 11.f, 1e-9f);
  EXPECT_NEAR(a.grad[3], 15.f, 1e-9f);

  // dB = A^T @ dC = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
  EXPECT_NEAR(b.grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(b.grad[1], 4.f, 1e-9f);
  EXPECT_NEAR(b.grad[2], 6.f, 1e-9f);
  EXPECT_NEAR(b.grad[3], 6.f, 1e-9f);
}

/// Verify ReLU activation forward filtering and gradient gating (subgradient).
TEST(grad_relu) {
  auto a = AG::Tensor<float>::from({3}, {-2.f, 0.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::relu(a);
  EXPECT_NEAR(c.data[0], 0.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 0.f, 1e-9f);
  EXPECT_NEAR(c.data[2], 3.f, 1e-9f);

  c.backward();
  EXPECT_NEAR(a.grad[0], 0.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 0.f, 1e-9f);
  EXPECT_NEAR(a.grad[2], 1.f, 1e-9f);
}

/// Verify 2D matrix transpose forward dimensions and gradient propagation.
TEST(grad_transpose) {
  auto a = AG::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, /*requires_grad=*/true);
  auto c = AG::transpose(a);
  EXPECT(c.shape[0] == 3 && c.shape[1] == 2);
  EXPECT_NEAR(c.data[0], 1.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 4.f, 1e-9f);
  EXPECT_NEAR(c.data[2], 2.f, 1e-9f);
  EXPECT_NEAR(c.data[3], 5.f, 1e-9f);
  EXPECT_NEAR(c.data[4], 3.f, 1e-9f);
  EXPECT_NEAR(c.data[5], 6.f, 1e-9f);

  c.backward();
  for (auto g : a.grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

/// Verify sum reduction forward value and broadcast gradient propagation.
TEST(grad_reduce) {
  auto a = AG::Tensor<float>::from({2, 2}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto c = AG::reduce(a);
  EXPECT_NEAR(c.data[0], 10.f, 1e-9f);

  c.backward();
  for (auto g : a.grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

/// Verify reverse-mode automatic differentiation across a multi-step chain DAG.
TEST(grad_dag_chain) {
  // a = [2], b = [3]
  // c = a + b = [5]
  // d = c * a = [10]
  // d(d)/da = c + a = 5 + 2 = 7
  // d(d)/db = a = 2
  auto a = AG::Tensor<float>::full({1}, 2.f, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto c = AG::add(a, b);
  auto d = AG::mul(c, a);
  EXPECT_NEAR(d.data[0], 10.f, 1e-9f);

  d.backward();
  EXPECT_NEAR(a.grad[0], 7.f, 1e-9f);
  EXPECT_NEAR(b.grad[0], 2.f, 1e-9f);
}

/// Verify gradient accumulation in a diamond graph where a variable is reused: y = x * x.
TEST(grad_diamond_graph) {
  // x = [3]
  // y = x * x = [9]
  // dy/dx = 2 * x = 6
  auto x = AG::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto y = AG::mul(x, x);
  EXPECT_NEAR(y.data[0], 9.f, 1e-9f);

  y.backward();
  EXPECT_NEAR(x.grad[0], 6.f, 1e-9f);
}

/// Verify that operands with requires_grad=false are not updated while gradient propagates.
TEST(grad_no_grad_operand) {
  auto a = AG::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/false);
  auto c = AG::mul(a, b);

  c.backward();
  EXPECT_NEAR(a.grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 5.f, 1e-9f);
  EXPECT(b.grad.empty());
}

// ===========================================================================
// Original Operator Overload Tests
// ===========================================================================

/// Verify element-wise operator+ overload.
TEST(operator_add) {
  auto a = AG::Tensor<float>::from({2}, {1.f, 2.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {3.f, 4.f}, /*requires_grad=*/true);
  auto c = a + b;
  EXPECT_NEAR(c.data[0], 4.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 6.f, 1e-9f);

  c.backward();
  EXPECT_NEAR(a.grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 1.f, 1e-9f);
  EXPECT_NEAR(b.grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(b.grad[1], 1.f, 1e-9f);
}

/// Verify element-wise operator* overload.
TEST(operator_mul) {
  auto a = AG::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/true);
  auto c = a * b;
  EXPECT_NEAR(c.data[0], 8.f, 1e-9f);
  EXPECT_NEAR(c.data[1], 15.f, 1e-9f);

  c.backward();
  EXPECT_NEAR(a.grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a.grad[1], 5.f, 1e-9f);
  EXPECT_NEAR(b.grad[0], 2.f, 1e-9f);
  EXPECT_NEAR(b.grad[1], 3.f, 1e-9f);
}

/// Verify expression pipeline using operator overloads: (a + b) * a.
TEST(operator_expression_pipeline) {
  auto a = AG::Tensor<float>::full({1}, 2.f, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto c = (a + b) * a;
  EXPECT_NEAR(c.data[0], 10.f, 1e-9f);

  c.backward();
  EXPECT_NEAR(a.grad[0], 7.f, 1e-9f);
  EXPECT_NEAR(b.grad[0], 2.f, 1e-9f);
}

/// Verify C++20 TensorScalar concept type constraints.
TEST(concept_tensor_scalar) {
  static_assert(AG::TensorScalar<float>);
  static_assert(AG::TensorScalar<const float>);
  static_assert(AG::TensorScalar<double>);
  static_assert(AG::TensorScalar<const double>);
  static_assert(AG::TensorScalar<int>);
  static_assert(AG::TensorScalar<const int>);

  static_assert(!AG::TensorScalar<char>);
  static_assert(!AG::TensorScalar<bool>);
  static_assert(!AG::TensorScalar<std::string>);
  EXPECT(true);
}

// ===========================================================================
// NEW: Arithmetic Operations
// ===========================================================================

/// Verify subtraction forward and gradients: da=+1, db=-1.
TEST(grad_sub) {
  auto a = AG::Tensor<float>::from({2}, {5.f, 7.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = AG::sub(a, b);
  EXPECT_NEAR(c.data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 5.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(a.grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(a.grad[1], 1.f, 1e-6f);
  EXPECT_NEAR(b.grad[0], -1.f, 1e-6f);
  EXPECT_NEAR(b.grad[1], -1.f, 1e-6f);
}

/// Verify negation forward and gradient: d(-x)/dx = -1.
TEST(grad_neg) {
  auto a = AG::Tensor<float>::from({2}, {3.f, -4.f}, /*requires_grad=*/true);
  auto c = AG::neg(a);
  EXPECT_NEAR(c.data[0], -3.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 4.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(a.grad[0], -1.f, 1e-6f);
  EXPECT_NEAR(a.grad[1], -1.f, 1e-6f);
}

/// Verify division forward and quotient rule gradients.
TEST(grad_div) {
  auto a = AG::Tensor<float>::from({2}, {6.f, 8.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = AG::div(a, b);
  EXPECT_NEAR(c.data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 4.f, 1e-6f);
  c.backward();
  // da = 1/b, db = -a/b^2
  EXPECT_NEAR(a.grad[0], 1.f / 3.f, 1e-5f);
  EXPECT_NEAR(a.grad[1], 1.f / 2.f, 1e-5f);
  EXPECT_NEAR(b.grad[0], -6.f / 9.f, 1e-5f);
  EXPECT_NEAR(b.grad[1], -8.f / 4.f, 1e-5f);
}

/// Verify power forward (x^3) and gradient (3*x^2).
TEST(grad_pow) {
  auto x = AG::Tensor<float>::from({1}, {2.f}, /*requires_grad=*/true);
  auto c = AG::pow(x, 3.f);
  EXPECT_NEAR(c.data[0], 8.f, 1e-5f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 12.f, 1e-4f);
}

/// Verify exponential forward and gradient: d/dx exp(x) = exp(x).
TEST(grad_exp) {
  auto x = AG::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto c = AG::exp(x);
  EXPECT_NEAR(c.data[0], 1.f, 1e-5f);
  EXPECT_NEAR(c.data[1], std::exp(1.f), 1e-5f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 1.f, 1e-5f);
  EXPECT_NEAR(x.grad[1], std::exp(1.f), 1e-4f);
}

/// Verify natural log forward and gradient: d/dx ln(x) = 1/x.
TEST(grad_log) {
  auto x = AG::Tensor<float>::from({2}, {1.f, std::exp(1.f)}, /*requires_grad=*/true);
  auto c = AG::log(x);
  EXPECT_NEAR(c.data[0], 0.f, 1e-5f);
  EXPECT_NEAR(c.data[1], 1.f, 1e-5f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 1.f, 1e-5f);
  EXPECT_NEAR(x.grad[1], 1.f / std::exp(1.f), 1e-5f);
}

// ===========================================================================
// NEW: Scalar-Tensor Operations
// ===========================================================================

/// Verify scalar addition forward and gradient: d/dx (x + s) = 1.
TEST(grad_add_scalar) {
  auto x = AG::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = AG::add_scalar(x, 10.f);
  EXPECT_NEAR(c.data[0], 13.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 15.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x.grad[1], 1.f, 1e-6f);
}

/// Verify scalar multiplication forward and gradient: d/dx (x * s) = s.
TEST(grad_mul_scalar) {
  auto x = AG::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = AG::mul_scalar(x, 4.f);
  EXPECT_NEAR(c.data[0], 12.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 20.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 4.f, 1e-6f);
  EXPECT_NEAR(x.grad[1], 4.f, 1e-6f);
}

// ===========================================================================
// NEW: Activation Functions
// ===========================================================================

/// Verify sigmoid forward and gradient: dσ/dx = σ(x)(1-σ(x)).
TEST(grad_sigmoid) {
  auto x = AG::Tensor<float>::from({2}, {0.f, 2.f}, /*requires_grad=*/true);
  auto c = AG::sigmoid(x);
  float s0 = 0.5f;
  float s1 = 1.f / (1.f + std::exp(-2.f));
  EXPECT_NEAR(c.data[0], s0, 1e-5f);
  EXPECT_NEAR(c.data[1], s1, 1e-5f);
  c.backward();
  EXPECT_NEAR(x.grad[0], s0 * (1 - s0), 1e-5f);
  EXPECT_NEAR(x.grad[1], s1 * (1 - s1), 1e-5f);
}

/// Verify tanh forward and gradient: d/dx tanh(x) = 1 - tanh²(x).
TEST(grad_tanh) {
  auto x = AG::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto c = AG::tanh_op(x);
  float t0 = std::tanh(0.f);
  float t1 = std::tanh(1.f);
  EXPECT_NEAR(c.data[0], t0, 1e-5f);
  EXPECT_NEAR(c.data[1], t1, 1e-5f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 1.f - t0 * t0, 1e-5f);
  EXPECT_NEAR(x.grad[1], 1.f - t1 * t1, 1e-5f);
}

/// Verify leaky ReLU forward and gradient with alpha.
TEST(grad_leaky_relu) {
  auto x = AG::Tensor<float>::from({3}, {-2.f, 0.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::leaky_relu(x, 0.01f);
  EXPECT_NEAR(c.data[0], -0.02f, 1e-6f);
  EXPECT_NEAR(c.data[1], 0.f, 1e-6f);
  EXPECT_NEAR(c.data[2], 3.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 0.01f, 1e-6f);
  EXPECT_NEAR(x.grad[1], 0.01f, 1e-6f);  // at 0, use alpha slope
  EXPECT_NEAR(x.grad[2], 1.f, 1e-6f);
}

/// Verify softmax output sums to 1 and preserves ordering.
TEST(forward_softmax) {
  auto x = AG::Tensor<float>::from({1, 3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::softmax(x, 1);
  float sum = c.data[0] + c.data[1] + c.data[2];
  EXPECT_NEAR(sum, 1.f, 1e-5f);
  EXPECT(c.data[2] > c.data[1]);
  EXPECT(c.data[1] > c.data[0]);
}

// ===========================================================================
// NEW: Reduction Operations
// ===========================================================================

/// Verify mean reduction forward and uniform gradient broadcast.
TEST(grad_mean) {
  auto x = AG::Tensor<float>::from({4}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto c = AG::mean(x);
  EXPECT_NEAR(c.data[0], 2.5f, 1e-6f);
  c.backward();
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(x.grad[i], 0.25f, 1e-6f);
  }
}

/// Verify max reduction forward and sparse gradient (only max element gets grad).
TEST(grad_max) {
  auto x = AG::Tensor<float>::from({4}, {1.f, 4.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::max(x);
  EXPECT_NEAR(c.data[0], 4.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(x.grad[0], 0.f, 1e-6f);
  EXPECT_NEAR(x.grad[1], 1.f, 1e-6f);
  EXPECT_NEAR(x.grad[2], 0.f, 1e-6f);
  EXPECT_NEAR(x.grad[3], 0.f, 1e-6f);
}

/// Verify sum_dim along dim=1 reduces columns and broadcasts gradient back.
TEST(grad_sum_dim) {
  auto x = AG::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                   /*requires_grad=*/true);
  auto c = AG::sum_dim(x, 1);
  EXPECT(c.shape[0] == 2 && c.shape[1] == 1);
  EXPECT_NEAR(c.data[0], 6.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 15.f, 1e-6f);
  c.backward();
  for (auto g : x.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

// ===========================================================================
// NEW: Shape / View Operations
// ===========================================================================

/// Verify reshape preserves data and passes gradient through.
TEST(grad_reshape) {
  auto x = AG::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                   /*requires_grad=*/true);
  auto c = AG::reshape(x, {3, 2});
  EXPECT(c.shape[0] == 3 && c.shape[1] == 2);
  EXPECT_NEAR(c.data[0], 1.f, 1e-6f);
  EXPECT_NEAR(c.data[5], 6.f, 1e-6f);
  c.backward();
  for (auto g : x.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/// Verify flatten converts to 1D and passes gradient through.
TEST(grad_flatten) {
  auto x = AG::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                   /*requires_grad=*/true);
  auto c = AG::flatten(x);
  EXPECT(c.shape.size() == 1);
  EXPECT(c.shape[0] == 6);
  c.backward();
  for (auto g : x.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/// Verify squeeze removes size-1 dimension.
TEST(grad_squeeze) {
  auto x = AG::Tensor<float>::from({1, 3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::squeeze(x, 0);
  EXPECT(c.shape.size() == 1 && c.shape[0] == 3);
  c.backward();
  for (auto g : x.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/// Verify unsqueeze inserts size-1 dimension.
TEST(grad_unsqueeze) {
  auto x = AG::Tensor<float>::from({3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::unsqueeze(x, 0);
  EXPECT(c.shape.size() == 2 && c.shape[0] == 1 && c.shape[1] == 3);
  c.backward();
  for (auto g : x.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/// Verify cat concatenates along dim=0 and splits gradients back.
TEST(grad_cat) {
  auto a = AG::Tensor<float>::from({2, 2}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2, 2}, {5.f, 6.f, 7.f, 8.f}, /*requires_grad=*/true);
  auto c = AG::cat({&a, &b}, 0);
  EXPECT(c.shape[0] == 4 && c.shape[1] == 2);
  EXPECT_NEAR(c.data[0], 1.f, 1e-6f);
  EXPECT_NEAR(c.data[4], 5.f, 1e-6f);
  c.backward();
  for (auto g : a.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
  for (auto g : b.grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

// ===========================================================================
// NEW: Loss Functions
// ===========================================================================

/// Verify MSE loss forward and gradient: d/dpred = 2*(pred-target)/n.
TEST(grad_mse_loss) {
  auto pred = AG::Tensor<float>::from({3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto target = AG::Tensor<float>::from({3}, {1.5f, 2.5f, 3.5f}, /*requires_grad=*/false);
  auto loss = AG::mse_loss(pred, target);
  // mse = mean([0.25, 0.25, 0.25]) = 0.25
  EXPECT_NEAR(loss.data[0], 0.25f, 1e-5f);
  loss.backward();
  // d_pred = 2*(pred-target)/n = 2*(-0.5)/3 = -1/3
  EXPECT_NEAR(pred.grad[0], -1.f / 3.f, 1e-5f);
  EXPECT_NEAR(pred.grad[1], -1.f / 3.f, 1e-5f);
  EXPECT_NEAR(pred.grad[2], -1.f / 3.f, 1e-5f);
}

/// Verify cross-entropy loss forward (positive loss) and gradient existence.
TEST(grad_cross_entropy) {
  // 2 samples, 3 classes
  auto logits =
      AG::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto targets = AG::Tensor<int>::from({2}, {2, 0});
  auto loss = AG::cross_entropy_loss(logits, targets);
  // loss should be positive
  EXPECT(loss.data[0] > 0.f);
  loss.backward();
  // gradients should exist and be non-zero
  EXPECT(logits.grad.size() == 6);
  // For sample 0 target=2: grad for class 2 should be negative (softmax[2]-1)/2
  // For sample 1 target=0: grad for class 0 should be negative (softmax[0]-1)/2
  // Check that at least some grads are non-zero
  bool has_nonzero = false;
  for (auto g : logits.grad) {
    if (std::fabs(g) > 1e-7f) has_nonzero = true;
  }
  EXPECT(has_nonzero);
}

// ===========================================================================
// NEW: Operator Overload Tests
// ===========================================================================

/// Verify operator- (binary subtraction).
TEST(operator_sub) {
  auto a = AG::Tensor<float>::from({2}, {5.f, 7.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = a - b;
  EXPECT_NEAR(c.data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 5.f, 1e-6f);
  c.backward();
  EXPECT_NEAR(a.grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(b.grad[0], -1.f, 1e-6f);
}

/// Verify unary operator- (negation).
TEST(operator_neg) {
  auto a = AG::Tensor<float>::from({2}, {3.f, -4.f}, /*requires_grad=*/true);
  auto c = -a;
  EXPECT_NEAR(c.data[0], -3.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 4.f, 1e-6f);
}

/// Verify operator/ (element-wise division).
TEST(operator_div) {
  auto a = AG::Tensor<float>::from({2}, {6.f, 8.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = a / b;
  EXPECT_NEAR(c.data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 4.f, 1e-6f);
}

/// Verify operator+ with scalar.
TEST(operator_add_scalar) {
  auto a = AG::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = a + 10.f;
  EXPECT_NEAR(c.data[0], 13.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 15.f, 1e-6f);
}

/// Verify operator* with scalar.
TEST(operator_mul_scalar) {
  auto a = AG::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = a * 4.f;
  EXPECT_NEAR(c.data[0], 12.f, 1e-6f);
  EXPECT_NEAR(c.data[1], 20.f, 1e-6f);
}

// ===========================================================================
// Runner
// ===========================================================================

/**
 * @brief Test runner entry point.
 *
 * Supports three invocation modes:
 * - No arguments: Runs all registered tests in sequence and summarizes results.
 * - `--list`: Prints all registered test names (one per line).
 * - `<test_name>`: Runs only the specified test case, exiting 0 on PASS, 1 on FAIL.
 */
int main(int argc, char* argv[]) {
  // If an argument is provided, support listing or running a specific test
  if (argc > 1) {
    std::string arg = argv[1];

    if (arg == "--list") {
      for (const auto& tc : g_tests) {
        std::printf("%s\n", tc.name);
      }
      return 0;
    }

    for (const auto& tc : g_tests) {
      if (arg == tc.name) {
        try {
          tc.fn();
          std::printf("  PASS  %s\n", tc.name);
          return 0;
        } catch (...) {
          std::printf("  FAIL  %s\n", tc.name);
          return 1;
        }
      }
    }

    std::fprintf(stderr, "Unknown test: %s\n", arg.c_str());
    return 1;
  }

  // Run all tests
  int passed = 0, failed = 0;
  for (auto& tc : g_tests) {
    try {
      tc.fn();
      std::printf("  PASS  %s\n", tc.name);
      ++passed;
    } catch (...) {
      std::printf("  FAIL  %s\n", tc.name);
      ++failed;
    }
  }
  std::printf("\n%d / %d tests passed\n", passed, passed + failed);
  return failed ? 1 : 0;
}
