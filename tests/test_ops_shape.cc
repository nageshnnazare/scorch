/**
 * @file test_ops_shape.cc
 * @brief Unit tests for tensor shape and view operations: reshape, flatten, squeeze, unsqueeze,
 * cat.
 *
 * @details
 * Validates forward reshaping and autograd backward gradient flow for:
 * - `reshape`: Dimension transformation with equal element count.
 * - `flatten`: 1-D vector flattening.
 * - `squeeze`: Singleton dimension removal.
 * - `unsqueeze`: Singleton dimension insertion.
 * - `cat`: Concatenation along specified axis.
 */

#include "test_common.h"

/**
 * @brief Tests tensor reshape and gradient pass-through.
 */
TEST(grad_reshape) {
  auto x = scorch::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                       /*requires_grad=*/true);
  auto c = scorch::reshape(x, {3, 2});
  EXPECT(c->shape[0] == 3 && c->shape[1] == 2);
  EXPECT_NEAR(c->data[0], 1.f, 1e-6f);
  EXPECT_NEAR(c->data[5], 6.f, 1e-6f);
  c->backward();
  for (auto g : x->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/**
 * @brief Tests tensor flattening to 1-D vector and gradient flow.
 */
TEST(grad_flatten) {
  auto x = scorch::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                       /*requires_grad=*/true);
  auto c = scorch::flatten(x);
  EXPECT(c->shape.size() == 1);
  EXPECT(c->shape[0] == 6);
  c->backward();
  for (auto g : x->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/**
 * @brief Tests squeeze removing a singleton dimension.
 */
TEST(grad_squeeze) {
  auto x = scorch::Tensor<float>::from({1, 3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::squeeze(x, 0);
  EXPECT(c->shape.size() == 1 && c->shape[0] == 3);
  c->backward();
  for (auto g : x->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/**
 * @brief Tests unsqueeze adding a singleton dimension at specified axis.
 */
TEST(grad_unsqueeze) {
  auto x = scorch::Tensor<float>::from({3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::unsqueeze(x, 0);
  EXPECT(c->shape.size() == 2 && c->shape[0] == 1 && c->shape[1] == 3);
  c->backward();
  for (auto g : x->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}

/**
 * @brief Tests tensor concatenation along dimension 0 and gradient splitting.
 */
TEST(grad_cat) {
  auto a = scorch::Tensor<float>::from({2, 2}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
  auto b = scorch::Tensor<float>::from({2, 2}, {5.f, 6.f, 7.f, 8.f}, /*requires_grad=*/true);
  auto c = scorch::cat({a, b}, 0);
  EXPECT(c->shape[0] == 4 && c->shape[1] == 2);
  EXPECT_NEAR(c->data[0], 1.f, 1e-6f);
  EXPECT_NEAR(c->data[4], 5.f, 1e-6f);
  c->backward();
  for (auto g : a->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
  for (auto g : b->grad) EXPECT_NEAR(g, 1.f, 1e-6f);
}
