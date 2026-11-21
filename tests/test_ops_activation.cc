/**
 * @file test_ops_activation.cc
 * @brief Unit tests for activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax.
 */

#include "test_common.h"

TEST(grad_relu) {
  auto a = AG::Tensor<float>::from({3}, {-2.f, 0.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::relu(a);
  EXPECT_NEAR(c->data[0], 0.f, 1e-9f);
  EXPECT_NEAR(c->data[1], 0.f, 1e-9f);
  EXPECT_NEAR(c->data[2], 3.f, 1e-9f);

  c->backward();
  EXPECT_NEAR(a->grad[0], 0.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 0.f, 1e-9f);
  EXPECT_NEAR(a->grad[2], 1.f, 1e-9f);
}

TEST(grad_leaky_relu) {
  auto x = AG::Tensor<float>::from({3}, {-2.f, 0.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::leaky_relu(x, 0.01f);
  EXPECT_NEAR(c->data[0], -0.02f, 1e-6f);
  EXPECT_NEAR(c->data[1], 0.f, 1e-6f);
  EXPECT_NEAR(c->data[2], 3.f, 1e-6f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 0.01f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 0.01f, 1e-6f);
  EXPECT_NEAR(x->grad[2], 1.f, 1e-6f);
}

TEST(grad_sigmoid) {
  auto x = AG::Tensor<float>::from({2}, {0.f, 2.f}, /*requires_grad=*/true);
  auto c = AG::sigmoid(x);
  float s0 = 0.5f;
  float s1 = 1.f / (1.f + std::exp(-2.f));
  EXPECT_NEAR(c->data[0], s0, 1e-5f);
  EXPECT_NEAR(c->data[1], s1, 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], s0 * (1 - s0), 1e-5f);
  EXPECT_NEAR(x->grad[1], s1 * (1 - s1), 1e-5f);
}

TEST(grad_tanh) {
  auto x = AG::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto c = AG::tanh_op(x);
  float t0 = std::tanh(0.f);
  float t1 = std::tanh(1.f);
  EXPECT_NEAR(c->data[0], t0, 1e-5f);
  EXPECT_NEAR(c->data[1], t1, 1e-5f);
  c->backward();
  EXPECT_NEAR(x->grad[0], 1.f - t0 * t0, 1e-5f);
  EXPECT_NEAR(x->grad[1], 1.f - t1 * t1, 1e-5f);
}

TEST(forward_softmax) {
  auto x = AG::Tensor<float>::from({1, 3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto c = AG::softmax(x, 1);
  float sum = c->data[0] + c->data[1] + c->data[2];
  EXPECT_NEAR(sum, 1.f, 1e-5f);
  EXPECT(c->data[2] > c->data[1]);
  EXPECT(c->data[1] > c->data[0]);
}
