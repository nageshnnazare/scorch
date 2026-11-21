/**
 * @file test_ops_trigonometric.cc
 * @brief Unit tests for differentiable trigonometric and mathematical operations.
 */

#include <cmath>
#include <numbers>

#include "ops/trigonometric.h"
#include "test_common.h"

TEST(grad_sin) {
  auto x = scorch::from({2}, {0.0f, static_cast<float>(std::numbers::pi / 2.0)},
                        /*requires_grad=*/true);
  auto y = scorch::sin(x);

  EXPECT_NEAR(y->data[0], 0.0f, 1e-5f);
  EXPECT_NEAR(y->data[1], 1.0f, 1e-5f);

  auto loss = scorch::reduce(y);
  loss->backward();

  // d/dx sin(0) = cos(0) = 1.0
  EXPECT_NEAR(x->grad[0], 1.0f, 1e-5f);
  // d/dx sin(pi/2) = cos(pi/2) = 0.0
  EXPECT_NEAR(x->grad[1], 0.0f, 1e-5f);
}

TEST(grad_cos) {
  auto x = scorch::from({2}, {0.0f, static_cast<float>(std::numbers::pi / 2.0)},
                        /*requires_grad=*/true);
  auto y = scorch::cos(x);

  EXPECT_NEAR(y->data[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y->data[1], 0.0f, 1e-5f);

  auto loss = scorch::reduce(y);
  loss->backward();

  // d/dx cos(0) = -sin(0) = 0.0
  EXPECT_NEAR(x->grad[0], 0.0f, 1e-5f);
  // d/dx cos(pi/2) = -sin(pi/2) = -1.0
  EXPECT_NEAR(x->grad[1], -1.0f, 1e-5f);
}

TEST(grad_tan) {
  auto x = scorch::from({1}, {0.0f}, /*requires_grad=*/true);
  auto y = scorch::tan(x);

  EXPECT_NEAR(y->data[0], 0.0f, 1e-5f);

  y->backward();
  // d/dx tan(0) = 1 / cos^2(0) = 1.0
  EXPECT_NEAR(x->grad[0], 1.0f, 1e-5f);
}

TEST(grad_asin) {
  auto x = scorch::from({1}, {0.5f}, /*requires_grad=*/true);
  auto y = scorch::asin(x);

  EXPECT_NEAR(y->data[0], static_cast<float>(std::asin(0.5)), 1e-5f);

  y->backward();
  // d/dx asin(0.5) = 1 / sqrt(1 - 0.25) = 1 / sqrt(0.75) = 1.1547005f
  float expected_grad = 1.0f / std::sqrt(1.0f - 0.25f);
  EXPECT_NEAR(x->grad[0], expected_grad, 1e-4f);
}

TEST(grad_acos) {
  auto x = scorch::from({1}, {0.5f}, /*requires_grad=*/true);
  auto y = scorch::acos(x);

  EXPECT_NEAR(y->data[0], static_cast<float>(std::acos(0.5)), 1e-5f);

  y->backward();
  // d/dx acos(0.5) = -1 / sqrt(1 - 0.25)
  float expected_grad = -1.0f / std::sqrt(1.0f - 0.25f);
  EXPECT_NEAR(x->grad[0], expected_grad, 1e-4f);
}

TEST(grad_atan) {
  auto x = scorch::from({1}, {1.0f}, /*requires_grad=*/true);
  auto y = scorch::atan(x);

  EXPECT_NEAR(y->data[0], static_cast<float>(std::numbers::pi / 4.0), 1e-5f);

  y->backward();
  // d/dx atan(1) = 1 / (1 + 1^2) = 0.5
  EXPECT_NEAR(x->grad[0], 0.5f, 1e-5f);
}

TEST(grad_sinh) {
  auto x = scorch::from({1}, {0.0f}, /*requires_grad=*/true);
  auto y = scorch::sinh(x);

  EXPECT_NEAR(y->data[0], 0.0f, 1e-5f);

  y->backward();
  // d/dx sinh(0) = cosh(0) = 1.0
  EXPECT_NEAR(x->grad[0], 1.0f, 1e-5f);
}

TEST(grad_cosh) {
  auto x = scorch::from({1}, {0.0f}, /*requires_grad=*/true);
  auto y = scorch::cosh(x);

  EXPECT_NEAR(y->data[0], 1.0f, 1e-5f);

  y->backward();
  // d/dx cosh(0) = sinh(0) = 0.0
  EXPECT_NEAR(x->grad[0], 0.0f, 1e-5f);
}

TEST(grad_tanh_alias) {
  auto x = scorch::from({1}, {0.0f}, /*requires_grad=*/true);
  auto y = scorch::tanh(x);

  EXPECT_NEAR(y->data[0], 0.0f, 1e-5f);

  y->backward();
  // d/dx tanh(0) = 1 - tanh^2(0) = 1.0
  EXPECT_NEAR(x->grad[0], 1.0f, 1e-5f);
}

TEST(grad_loge) {
  auto x = scorch::from({1}, {static_cast<float>(std::numbers::e)}, /*requires_grad=*/true);
  auto y = scorch::loge(x);

  EXPECT_NEAR(y->data[0], 1.0f, 1e-5f);

  y->backward();
  // d/dx ln(e) = 1 / e
  EXPECT_NEAR(x->grad[0], static_cast<float>(1.0 / std::numbers::e), 1e-5f);
}

TEST(grad_log2) {
  auto x = scorch::from({1}, {4.0f}, /*requires_grad=*/true);
  auto y = scorch::log2(x);

  EXPECT_NEAR(y->data[0], 2.0f, 1e-5f);

  y->backward();
  // d/dx log2(4) = 1 / (4 * ln 2)
  float expected_grad = 1.0f / (4.0f * std::log(2.0f));
  EXPECT_NEAR(x->grad[0], expected_grad, 1e-4f);
}

TEST(grad_log10) {
  auto x = scorch::from({1}, {100.0f}, /*requires_grad=*/true);
  auto y = scorch::log10(x);

  EXPECT_NEAR(y->data[0], 2.0f, 1e-5f);

  y->backward();
  // d/dx log10(100) = 1 / (100 * ln 10)
  float expected_grad = 1.0f / (100.0f * std::log(10.0f));
  EXPECT_NEAR(x->grad[0], expected_grad, 1e-5f);
}

TEST(grad_sqrt) {
  auto x = scorch::from({1}, {4.0f}, /*requires_grad=*/true);
  auto y = scorch::sqrt(x);

  EXPECT_NEAR(y->data[0], 2.0f, 1e-5f);

  y->backward();
  // d/dx sqrt(4) = 1 / (2 * sqrt(4)) = 0.25
  EXPECT_NEAR(x->grad[0], 0.25f, 1e-5f);
}

TEST(grad_rsqrt) {
  auto x = scorch::from({1}, {4.0f}, /*requires_grad=*/true);
  auto y = scorch::rsqrt(x);

  EXPECT_NEAR(y->data[0], 0.5f, 1e-5f);

  y->backward();
  // d/dx (x^{-1/2}) = -0.5 * x^{-3/2} = -0.5 * (1/8) = -0.0625
  EXPECT_NEAR(x->grad[0], -0.0625f, 1e-5f);
}

TEST(grad_abs) {
  auto x = scorch::from({2}, {-3.0f, 5.0f}, /*requires_grad=*/true);
  auto y = scorch::abs(x);

  EXPECT_NEAR(y->data[0], 3.0f, 1e-5f);
  EXPECT_NEAR(y->data[1], 5.0f, 1e-5f);

  auto loss = scorch::reduce(y);
  loss->backward();

  EXPECT_NEAR(x->grad[0], -1.0f, 1e-5f);
  EXPECT_NEAR(x->grad[1], 1.0f, 1e-5f);
}
