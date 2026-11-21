/**
 * @file test_ops_arithmetic.cc
 * @brief Unit tests for arithmetic operations, scalar operations, and operator overloads.
 */

#include "test_common.h"

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

TEST(grad_sub) {
  auto a = scorch::Tensor<float>::from({2}, {5.f, 7.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = scorch::sub(a, b);
  EXPECT_NEAR(c->data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 5.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(a->grad[1], 1.f, 1e-6f);
  EXPECT_NEAR(b->grad[0], -1.f, 1e-6f);
  EXPECT_NEAR(b->grad[1], -1.f, 1e-6f);
}

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

TEST(grad_div) {
  auto a = scorch::Tensor<float>::from({2}, {6.f, 8.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = scorch::div(a, b);
  EXPECT_NEAR(c->data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f / 3.f, 1e-5f);
  EXPECT_NEAR(a->grad[1], 1.f / 2.f, 1e-5f);
  EXPECT_NEAR(b->grad[0], -6.f / 9.f, 1e-5f);
  EXPECT_NEAR(b->grad[1], -8.f / 4.f, 1e-5f);
}

TEST(grad_neg) {
  auto a = scorch::Tensor<float>::from({2}, {3.f, -4.f}, /*requires_grad=*/true);
  auto c = scorch::neg(a);
  EXPECT_NEAR(c->data[0], -3.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(a->grad[0], -1.f, 1e-6f);
  EXPECT_NEAR(a->grad[1], -1.f, 1e-6f);
}

TEST(grad_pow) {
  auto x = scorch::Tensor<float>::from({1}, {2.f}, /*requires_grad=*/true);
  auto c = scorch::pow(x, 3.f);
  EXPECT_NEAR(c->data[0], 8.f, 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 12.f, 1e-4f);
}

TEST(grad_exp) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto c = scorch::exp(x);
  EXPECT_NEAR(c->data[0], 1.f, 1e-5f);
  EXPECT_NEAR(c->data[1], std::exp(1.f), 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-5f);
  EXPECT_NEAR(x->grad[1], std::exp(1.f), 1e-4f);
}

TEST(grad_log) {
  auto x = scorch::Tensor<float>::from({2}, {1.f, std::exp(1.f)}, /*requires_grad=*/true);
  auto c = scorch::log(x);
  EXPECT_NEAR(c->data[0], 0.f, 1e-5f);
  EXPECT_NEAR(c->data[1], 1.f, 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-5f);
  EXPECT_NEAR(x->grad[1], 1.f / std::exp(1.f), 1e-5f);
}

TEST(grad_add_scalar) {
  auto x = scorch::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = scorch::add_scalar(x, 10.f);
  EXPECT_NEAR(c->data[0], 13.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 15.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 1.f, 1e-6f);
}

TEST(grad_mul_scalar) {
  auto x = scorch::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = scorch::mul_scalar(x, 4.f);
  EXPECT_NEAR(c->data[0], 12.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 20.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 4.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 4.f, 1e-6f);
}

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

TEST(operator_sub) {
  auto a = scorch::Tensor<float>::from({2}, {5.f, 7.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = a - b;
  EXPECT_NEAR(c->data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 5.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(a->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(b->grad[0], -1.f, 1e-6f);
}

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

TEST(operator_div) {
  auto a = scorch::Tensor<float>::from({2}, {6.f, 8.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2}, {3.f, 2.f}, /*requires_grad=*/true);
  auto c = a / b;
  EXPECT_NEAR(c->data[0], 2.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-6f);
}

TEST(operator_neg) {
  auto a = scorch::Tensor<float>::from({2}, {3.f, -4.f}, /*requires_grad=*/true);
  auto c = -a;
  EXPECT_NEAR(c->data[0], -3.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-6f);
}

TEST(operator_add_scalar) {
  auto a = scorch::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = a + 10.f;
  EXPECT_NEAR(c->data[0], 13.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 15.f, 1e-6f);
}

TEST(operator_mul_scalar) {
  auto a = scorch::Tensor<float>::from({2}, {3.f, 5.f}, /*requires_grad=*/true);
  auto c = a * 4.f;
  EXPECT_NEAR(c->data[0], 12.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 20.f, 1e-6f);
}

TEST(operator_expression_pipeline) {
  auto a = scorch::Tensor<float>::full({1}, 2.f, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto c = (a + b) * a;
  EXPECT_NEAR(c->data[0], 10.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 7.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 2.f, 1e-9f);
}
