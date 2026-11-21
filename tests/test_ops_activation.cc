/**
 * @file test_ops_activation.cc
 * @brief Unit tests for activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax.
 *
 * @details
 * Validates forward evaluation and analytical backward gradients for:
 * - `relu`: Rectified Linear Unit.
 * - `leaky_relu`: Leaky ReLU with custom slope.
 * - `sigmoid`: Logistic sigmoid function.
 * - `tanh_op`: Hyperbolic tangent function.
 * - `softmax`: Softmax probability normalization.
 */

#include "test_common.h"

/**
 * @brief Tests forward clamp and subgradient of ReLU.
 */
TEST(grad_relu) {
  auto a = scorch::Tensor<float>::from({3}, {-2.f, 0.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::relu(a);
  EXPECT_NEAR(c->data[0], 0.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 0.f, 1e-9f);
  EXPECT_NEAR(c->data[2], 3.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 0.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 0.f, 1e-9f);
  EXPECT_NEAR(a->grad[2], 1.f, 1e-9f);
}

/**
 * @brief Tests forward scaling and slope gradient of Leaky ReLU.
 */
TEST(grad_leaky_relu) {
  auto x = scorch::Tensor<float>::from({3}, {-2.f, 0.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::leaky_relu(x, 0.01f);
  EXPECT_NEAR(c->data[0], -0.02f, 1e-6f);
  EXPECT_NEAR(c->data[1], 0.f, 1e-6f);
  EXPECT_NEAR(c->data[2], 3.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 0.01f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 0.01f, 1e-6f);
  EXPECT_NEAR(x->grad[2], 1.f, 1e-6f);
}

/**
 * @brief Tests forward sigmoid values and derivative rule s * (1 - s).
 */
TEST(grad_sigmoid) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 2.f}, /*requires_grad=*/true);
  auto c = scorch::sigmoid(x);
  float s0 = 0.5f;
  float s1 = 1.f / (1.f + std::exp(-2.f));
  EXPECT_NEAR(c->data[0], s0, 1e-5f);
  EXPECT_NEAR(c->data[1], s1, 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], s0 * (1 - s0), 1e-5f);
  EXPECT_NEAR(x->grad[1], s1 * (1 - s1), 1e-5f);
}

/**
 * @brief Tests forward tanh values and derivative rule 1 - tanh^2.
 */
TEST(grad_tanh) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto c = scorch::tanh_op(x);
  float t0 = std::tanh(0.f);
  float t1 = std::tanh(1.f);
  EXPECT_NEAR(c->data[0], t0, 1e-5f);
  EXPECT_NEAR(c->data[1], t1, 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 1.f - t0 * t0, 1e-5f);
  EXPECT_NEAR(x->grad[1], 1.f - t1 * t1, 1e-5f);
}

/**
 * @brief Tests forward softmax probability distribution summing to 1.
 */
TEST(forward_softmax) {
  auto x = scorch::Tensor<float>::from({1, 3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = scorch::softmax(x, 1);
  float sum = c->data[0] + c->data[1] + c->data[2];
  EXPECT_NEAR(sum, 1.f, 1e-5f);
  EXPECT(c->data[2] > c->data[1]);
  EXPECT(c->data[1] > c->data[0]);
}
