/**
 * @file test_ops_arithmetic.cc
 * @brief Unit tests for arithmetic operations, scalar operations, and operator overloads.
 *
 * @details
 * Validates forward evaluations and analytical gradients for:
 * - Free functions: `add`, `sub`, `mul`, `div`, `neg`, `pow`, `exp`, `log`, `add_scalar`,
 * `mul_scalar`.
 * - C++ operator overloads: `+`, `-`, `*`, `/`, unary `-`.
 * - Complex composite operator expression pipelines.
 */

#include "test_common.h"

/**
 * @brief Tests element-wise tensor addition gradients.
 */
TEST(grad_add) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/true);
  auto c = scorch::add(a, b);
  EXPECT_NEAR(c->data[0], 6.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 8.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 1.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(b->grad[1], 1.f, 1e-9f);
}

/**
 * @brief Tests element-wise tensor subtraction gradients.
 */
TEST(grad_sub) {
  auto a = scorch::Tensor<float>::from({2}, {5.f, 7.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::sub(a, b);
  EXPECT_NEAR(c->data[0], 3.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 1.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], -1.f, 1e-9f);
  EXPECT_NEAR(b->grad[1], -1.f, 1e-9f);
}

/**
 * @brief Tests element-wise tensor multiplication gradients.
 */
TEST(grad_mul) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/true);
  auto c = scorch::mul(a, b);
  EXPECT_NEAR(c->data[0], 8.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 15.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 5.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 2.f, 1e-9f);
  EXPECT_NEAR(b->grad[1], 3.f, 1e-9f);
}

/**
 * @brief Tests element-wise tensor division gradients.
 */
TEST(grad_div) {
  auto a = scorch::Tensor<float>::from({2}, {6.f, 8.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {2.f, 4.f}, /*requires_grad=*/true);
  auto c = scorch::div(a, b);
  EXPECT_NEAR(c->data[0], 3.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 2.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 0.5f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 0.25f, 1e-9f);
  EXPECT_NEAR(b->grad[0], -1.5f, 1e-9f);
  EXPECT_NEAR(b->grad[1], -0.5f, 1e-9f);
}

/**
 * @brief Tests element-wise negation gradients.
 */
TEST(grad_neg) {
  auto a = scorch::Tensor<float>::from({2}, {3.f, -5.f}, /*requires_grad=*/true);
  auto c = scorch::neg(a);
  EXPECT_NEAR(c->data[0], -3.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 5.f, 1e-9f);
  c->backward();
  EXPECT_NEAR(a->grad[0], -1.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], -1.f, 1e-9f);
}

/**
 * @brief Tests element-wise power function gradients.
 */
TEST(grad_pow) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::pow(a, 3.f);
  EXPECT_NEAR(c->data[0], 8.f, 1e-5f);
  EXPECT_NEAR(c->data[1], 27.f, 1e-5f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 12.f, 1e-5f);
  EXPECT_NEAR(a->grad[1], 27.f, 1e-5f);
}

/**
 * @brief Tests natural exponential function gradients.
 */
TEST(grad_exp) {
  auto a = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto c = scorch::exp(a);
  EXPECT_NEAR(c->data[0], 1.f, 1e-5f);
  EXPECT_NEAR(c->data[1], static_cast<float>(M_E), 1e-5f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-5f);
  EXPECT_NEAR(a->grad[1], static_cast<float>(M_E), 1e-5f);
}

/**
 * @brief Tests natural logarithm function gradients.
 */
TEST(grad_log) {
  auto a = scorch::Tensor<float>::from({2}, {1.f, static_cast<float>(M_E)}, /*requires_grad=*/true);
  auto c = scorch::log(a);
  EXPECT_NEAR(c->data[0], 0.f, 1e-5f);
  EXPECT_NEAR(c->data[1], 1.f, 1e-5f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-5f);
  EXPECT_NEAR(a->grad[1], 1.f / static_cast<float>(M_E), 1e-5f);
}

/**
 * @brief Tests scalar addition gradients.
 */
TEST(grad_add_scalar) {
  auto a = scorch::Tensor<float>::from({2}, {1.f, 2.f}, /*requires_grad=*/true);
  auto c = scorch::add_scalar(a, 5.f);
  EXPECT_NEAR(c->data[0], 6.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 7.f, 1e-9f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 1.f, 1e-9f);
}

/**
 * @brief Tests scalar multiplication gradients.
 */
TEST(grad_mul_scalar) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::mul_scalar(a, 4.f);
  EXPECT_NEAR(c->data[0], 8.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 12.f, 1e-9f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 4.f, 1e-9f);
}

/**
 * @brief Tests operator+ overload between tensors.
 */
TEST(operator_add) {
  auto a = scorch::Tensor<float>::from({2}, {1.f, 2.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {3.f, 4.f}, /*requires_grad=*/true);
  auto c = a + b;
  EXPECT_NEAR(c->data[0], 4.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 6.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 1.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(b->grad[1], 1.f, 1e-9f);
}

/**
 * @brief Tests operator- overload between tensors.
 */
TEST(operator_sub) {
  auto a = scorch::Tensor<float>::from({2}, {5.f, 7.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto c = a - b;
  EXPECT_NEAR(c->data[0], 3.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-9f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], -1.f, 1e-9f);
}

/**
 * @brief Tests operator* overload between tensors.
 */
TEST(operator_mul) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/true);
  auto c = a * b;
  EXPECT_NEAR(c->data[0], 8.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 15.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 5.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 2.f, 1e-9f);
  EXPECT_NEAR(b->grad[1], 3.f, 1e-9f);
}

/**
 * @brief Tests operator/ overload between tensors.
 */
TEST(operator_div) {
  auto a = scorch::Tensor<float>::from({2}, {6.f, 8.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {2.f, 4.f}, /*requires_grad=*/true);
  auto c = a / b;
  EXPECT_NEAR(c->data[0], 3.f, 1e-9f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 0.5f, 1e-9f);
}

/**
 * @brief Tests unary operator- overload.
 */
TEST(operator_neg) {
  auto a = scorch::Tensor<float>::from({2}, {3.f, -5.f}, /*requires_grad=*/true);
  auto c = -a;
  EXPECT_NEAR(c->data[0], -3.f, 1e-9f);
  c->backward();
  EXPECT_NEAR(a->grad[0], -1.f, 1e-9f);
}

/**
 * @brief Tests operator+ overload with scalar operand.
 */
TEST(operator_add_scalar) {
  auto a = scorch::Tensor<float>::from({2}, {1.f, 2.f}, /*requires_grad=*/true);
  auto c = a + 5.f;
  EXPECT_NEAR(c->data[0], 6.f, 1e-9f);
}

/**
 * @brief Tests operator* overload with scalar operand.
 */
TEST(operator_mul_scalar) {
  auto a = scorch::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto c = 4.f * a;
  EXPECT_NEAR(c->data[0], 8.f, 1e-9f);
}

/**
 * @brief Tests complex chained arithmetic pipeline (w + b) * w.
 */
TEST(operator_expression_pipeline) {
  auto w = scorch::Tensor<float>::ones({2, 3}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::full({2, 3}, 2.f, /*requires_grad=*/true);

  // c = (w + b) * w = (1 + 2) * 1 = 3
  auto c = (w + b) * w;
  for (auto v : c->data) EXPECT_NEAR(v, 3.f, 1e-9f);

  // dc/dw = 2w + b = 2*1 + 2 = 4
  // dc/db = w = 1
  c->backward();
  for (auto g : w->grad) EXPECT_NEAR(g, 4.f, 1e-9f);
  for (auto g : b->grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}
