/**
 * @file test_ops_reduction.cc
 * @brief Unit tests for reduction operations: sum, mean, max, sum_dim.
 */

#include "test_common.h"

TEST(grad_reduce) {
  auto a = scorch::Tensor<float>::from({2, 2}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto c = scorch::reduce(a);
  EXPECT_NEAR(c->data[0], 10.f, 1e-9f);

  c->backward();
  for (auto g : a->grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

TEST(grad_mean) {
  auto x = scorch::Tensor<float>::from({4}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto c = scorch::mean(x);
  EXPECT_NEAR(c->data[0], 2.5f, 1e-6f);
  c->backward();
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(x->grad[i], 0.25f, 1e-6f);
  }
}

TEST(grad_max) {
  auto x = scorch::Tensor<float>::from({4}, {1.f, 4.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::max(x);
  EXPECT_NEAR(c->data[0], 4.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 0.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[2], 0.f, 1e-6f);
  EXPECT_NEAR(x->grad[3], 0.f, 1e-6f);
}

TEST(grad_sum_dim) {
  auto x = scorch::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                       /*requires_grad=*/true);
  auto c = scorch::sum_dim(x, 1);
  EXPECT(c->shape[0] == 2 && c->shape[1] == 1);
  EXPECT_NEAR(c->data[0], 6.f, 1e-6f);
  EXPECT_NEAR(c->data[1], 15.f, 1e-6f);
  c->backward();
  for (auto g : x->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}
