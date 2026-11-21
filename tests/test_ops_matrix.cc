/**
 * @file test_ops_matrix.cc
 * @brief Unit tests for 2D matrix multiplication and transposition.
 *
 * @details
 * Validates forward calculations and autograd backward derivatives for:
 * - `matmul`: 2D matrix product.
 * - `transpose`: Matrix transposition.
 */

#include "test_common.h"

/**
 * @brief Tests forward output and gradients of 2D matrix multiplication.
 */
TEST(grad_matmul) {
  auto a = scorch::Tensor<float>::from({2, 2}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2, 2}, {5.f, 6.f, 7.f, 8.f}, /*requires_grad=*/true);
  auto c = scorch::matmul(a, b);
  EXPECT_NEAR(c->data[0], 19.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 22.f, 1e-9f);
  EXPECT_NEAR(c->data[2], 43.f, 1e-9f);
  EXPECT_NEAR(c->data[3], 50.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 11.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 15.f, 1e-9f);
  EXPECT_NEAR(a->grad[2], 11.f, 1e-9f);
  EXPECT_NEAR(a->grad[3], 15.f, 1e-9f);

  EXPECT_NEAR(b->grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(b->grad[1], 4.f, 1e-9f);
  EXPECT_NEAR(b->grad[2], 6.f, 1e-9f);
  EXPECT_NEAR(b->grad[3], 6.f, 1e-9f);
}

/**
 * @brief Tests forward output and gradients of 2D matrix transposition.
 */
TEST(grad_transpose) {
  auto a =
      scorch::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, /*requires_grad=*/true);
  auto c = scorch::transpose(a);
  EXPECT(c->shape[0] == 3 && c->shape[1] == 2);
  EXPECT_NEAR(c->data[0], 1.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 4.f, 1e-9f);
  EXPECT_NEAR(c->data[2], 2.f, 1e-9f);
  EXPECT_NEAR(c->data[3], 5.f, 1e-9f);
  EXPECT_NEAR(c->data[4], 3.f, 1e-9f);
  EXPECT_NEAR(c->data[5], 6.f, 1e-9f);

  c->backward();
  for (auto g : a->grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}
