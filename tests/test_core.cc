/**
 * @file test_core.cc
 * @brief Unit tests for core Tensor construction, indexing, and DAG traversal.
 *
 * @details
 * Validates:
 * - Tensor allocation factories (`zeros`, `ones`, `full`, `rand`, `from`).
 * - Multidimensional bracket indexing and element mutation (`t[i][j]`).
 * - Shape calculation utilities (`numel`, `flatid`).
 * - Backward gradient propagation across sequential chains and diamond graphs.
 * - C++20 `TensorScalar` concept constraints.
 */

#include "test_common.h"

/**
 * @brief Tests shape and element count for zeros factory.
 */
TEST(zeros_shape) {
  auto t = scorch::Tensor<float>::zeros({3, 5});
  EXPECT(t->shape.size() == 2);
  EXPECT(t->shape[0] == 3);
  EXPECT(t->shape[1] == 5);
  EXPECT(t->data.size() == 15);
}

/**
 * @brief Tests all values are initialized to 0 in zeros factory.
 */
TEST(zeros_values) {
  auto t = scorch::Tensor<float>::zeros({2, 4});
  for (auto v : t->data) EXPECT_NEAR(v, 0.f, 1e-9f);
}

/**
 * @brief Tests all values are initialized to 1 in ones factory.
 */
TEST(ones_values) {
  auto t = scorch::Tensor<float>::ones({2, 3});
  for (auto v : t->data) EXPECT_NEAR(v, 1.f, 1e-9f);
}

/**
 * @brief Tests all values are initialized to the requested constant in full factory.
 */
TEST(full_values) {
  auto t = scorch::Tensor<float>::full({3, 3}, 7.f);
  for (auto v : t->data) EXPECT_NEAR(v, 7.f, 1e-9f);
}

/**
 * @brief Tests that rand generates values strictly within [0, 1).
 */
TEST(rand_range) {
  auto t = scorch::Tensor<float>::rand({10, 10});
  for (auto v : t->data) {
    EXPECT(v >= 0.f);
    EXPECT(v < 1.f);
  }
}

/**
 * @brief Tests initialization from existing std::vector buffer.
 */
TEST(from_happy) {
  auto t = scorch::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT(t->shape[0] == 2 && t->shape[1] == 3);
  EXPECT_NEAR(t->data[0], 1.f, 1e-9f);
  EXPECT_NEAR(t->data[5], 6.f, 1e-9f);
}

/**
 * @brief Tests that from throws invalid_argument when buffer size != numel(shape).
 */
TEST(from_mismatch_throws) {
  bool threw = false;
  try {
    scorch::Tensor<float>::from({2, 3}, {1, 2, 3});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

/**
 * @brief Tests multidimensional bracket indexing for reading elements.
 */
TEST(indexing_read) {
  auto t = scorch::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_NEAR(static_cast<float>((*t)[0][0]), 1.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[0][2]), 3.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[1][0]), 4.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[1][2]), 6.f, 1e-9f);
}

/**
 * @brief Tests multidimensional bracket indexing for writing elements.
 */
TEST(indexing_write) {
  auto t = scorch::Tensor<float>::zeros({2, 3});
  (*t)[1][2] = 42.f;
  EXPECT_NEAR(static_cast<float>((*t)[1][2]), 42.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[1][1]), 0.f, 1e-9f);
}

/**
 * @brief Tests requires_grad property setting on tensors.
 */
TEST(requires_grad_flag) {
  auto a = scorch::Tensor<float>::zeros({2}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::zeros({2}, /*requires_grad=*/false);
  EXPECT(a->requires_grad == true);
  EXPECT(b->requires_grad == false);
}

/**
 * @brief Tests numel and flatid index mapping calculations.
 */
TEST(size_and_elements) {
  scorch::Shape shape{2, 3};
  auto t = scorch::Tensor<float>::from(shape, {1, 2, 3, 4, 5, 6});
  EXPECT(scorch::numel(shape) == 6);
  EXPECT(scorch::flatid(shape, {1, 1}) == 4);
  EXPECT(t->data[scorch::flatid(shape, {1, 2})] == 6);
}

/**
 * @brief Tests calling backward directly on a leaf tensor seeds gradient 1.0.
 */
TEST(backward_leaf) {
  auto t = scorch::Tensor<float>::zeros({2, 3}, /*requires_grad=*/true);
  t->backward();
  EXPECT(t->grad.size() == 6);
  for (auto g : t->grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

/**
 * @brief Tests reverse autodiff through a chained DAG (d = (a + b) * a).
 */
TEST(grad_dag_chain) {
  auto a = scorch::Tensor<float>::full({1}, 2.f, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto c = scorch::add(a, b);
  auto d = scorch::mul(c, a);
  EXPECT_NEAR(d->data[0], 10.f, 1e-9f);

  d->backward();
  EXPECT_NEAR(a->grad[0], 7.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 2.f, 1e-9f);
}

/**
 * @brief Tests gradient accumulation in diamond graph topology (y = x * x).
 */
TEST(grad_diamond_graph) {
  auto x = scorch::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto y = scorch::mul(x, x);
  EXPECT_NEAR(y->data[0], 9.f, 1e-9f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 6.f, 1e-9f);
}

/**
 * @brief Tests gradient propagation when one operand does not require grad.
 */
TEST(grad_no_grad_operand) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/false);
  auto c = scorch::mul(a, b);

  c->backward();
  EXPECT_NEAR(a->grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 5.f, 1e-9f);
  EXPECT(b->grad.empty());
}

/**
 * @brief Tests C++20 TensorScalar concept constraints at compile time.
 */
TEST(concept_tensor_scalar) {
  static_assert(scorch::TensorScalar<float>);
  static_assert(scorch::TensorScalar<const float>);
  static_assert(scorch::TensorScalar<double>);
  static_assert(scorch::TensorScalar<const double>);
  static_assert(scorch::TensorScalar<int>);
  static_assert(scorch::TensorScalar<const int>);

  static_assert(!scorch::TensorScalar<char>);
  static_assert(!scorch::TensorScalar<bool>);
  static_assert(!scorch::TensorScalar<std::string>);
  EXPECT(true);
}
