/**
 * @file test_ops_trigonometric.cc
 * @brief Unit tests for differentiable trigonometric and mathematical operations.
 *
 * @details
 * Validates forward values and analytical derivatives for:
 * - Circular: `sin`, `cos`, `tan`
 * - Inverse: `asin`, `acos`, `atan`
 * - Hyperbolic: `sinh`, `cosh`, `tanh_alias`
 * - Logarithms: `loge`, `log2`, `log10`
 * - Roots & absolute value: `sqrt`, `rsqrt`, `abs`
 */

#include <cmath>
#include <numbers>

#include "ops/trigonometric.h"
#include "test_common.h"

/**
 * @brief Tests forward evaluation and cos gradient of sin.
 */
TEST(grad_sin) {
  auto x = scorch::Tensor<float>::from(
      {3}, {0.f, static_cast<float>(std::numbers::pi) / 2.f, static_cast<float>(std::numbers::pi)},
      /*requires_grad=*/true);
  auto y = scorch::sin(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 1.f, 1e-6f);
  EXPECT_NEAR(y->data[2], 0.f, 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 0.f, 1e-6f);
  EXPECT_NEAR(x->grad[2], -1.f, 1e-6f);
}

/**
 * @brief Tests forward evaluation and -sin gradient of cos.
 */
TEST(grad_cos) {
  auto x = scorch::Tensor<float>::from(
      {3}, {0.f, static_cast<float>(std::numbers::pi) / 2.f, static_cast<float>(std::numbers::pi)},
      /*requires_grad=*/true);
  auto y = scorch::cos(x);
  EXPECT_NEAR(y->data[0], 1.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[2], -1.f, 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 0.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], -1.f, 1e-6f);
  EXPECT_NEAR(x->grad[2], 0.f, 1e-6f);
}

/**
 * @brief Tests forward evaluation and sec^2 gradient of tan.
 */
TEST(grad_tan) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, static_cast<float>(std::numbers::pi) / 4.f},
                                       /*requires_grad=*/true);
  auto y = scorch::tan(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 1.f, 1e-5f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 2.f, 1e-5f);
}

/**
 * @brief Tests forward evaluation and 1/sqrt(1-x^2) gradient of asin.
 */
TEST(grad_asin) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 0.5f}, /*requires_grad=*/true);
  auto y = scorch::asin(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::asin(0.5)), 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 1.f / std::sqrt(1.f - 0.25f), 1e-5f);
}

/**
 * @brief Tests forward evaluation and -1/sqrt(1-x^2) gradient of acos.
 */
TEST(grad_acos) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 0.5f}, /*requires_grad=*/true);
  auto y = scorch::acos(x);
  EXPECT_NEAR(y->data[0], static_cast<float>(std::numbers::pi) / 2.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::acos(0.5)), 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], -1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], -1.f / std::sqrt(1.f - 0.25f), 1e-5f);
}

/**
 * @brief Tests forward evaluation and 1/(1+x^2) gradient of atan.
 */
TEST(grad_atan) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto y = scorch::atan(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::numbers::pi) / 4.f, 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 0.5f, 1e-6f);
}

/**
 * @brief Tests forward evaluation and cosh gradient of sinh.
 */
TEST(grad_sinh) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto y = scorch::sinh(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::sinh(1.0)), 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], static_cast<float>(std::cosh(1.0)), 1e-6f);
}

/**
 * @brief Tests forward evaluation and sinh gradient of cosh.
 */
TEST(grad_cosh) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto y = scorch::cosh(x);
  EXPECT_NEAR(y->data[0], 1.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::cosh(1.0)), 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 0.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], static_cast<float>(std::sinh(1.0)), 1e-6f);
}

/**
 * @brief Tests tanh alias convenience wrapper.
 */
TEST(grad_tanh_alias) {
  auto x = scorch::Tensor<float>::from({2}, {0.f, 1.f}, /*requires_grad=*/true);
  auto y = scorch::tanh(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::tanh(1.0)), 1e-6f);

  y->backward();
  float dt = 1.f - static_cast<float>(std::tanh(1.0) * std::tanh(1.0));
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], dt, 1e-6f);
}

/**
 * @brief Tests natural logarithm loge alias.
 */
TEST(grad_loge) {
  auto x = scorch::Tensor<float>::from({2}, {1.f, 2.f}, /*requires_grad=*/true);
  auto y = scorch::loge(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], static_cast<float>(std::log(2.0)), 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 0.5f, 1e-6f);
}

/**
 * @brief Tests base-2 logarithm forward values and gradients.
 */
TEST(grad_log2) {
  auto x = scorch::Tensor<float>::from({2}, {1.f, 2.f}, /*requires_grad=*/true);
  auto y = scorch::log2(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 1.f, 1e-6f);

  y->backward();
  float ln2 = static_cast<float>(std::numbers::ln2);
  EXPECT_NEAR(x->grad[0], 1.f / ln2, 1e-5f);
  EXPECT_NEAR(x->grad[1], 1.f / (2.f * ln2), 1e-5f);
}

/**
 * @brief Tests base-10 logarithm forward values and gradients.
 */
TEST(grad_log10) {
  auto x = scorch::Tensor<float>::from({2}, {1.f, 10.f}, /*requires_grad=*/true);
  auto y = scorch::log10(x);
  EXPECT_NEAR(y->data[0], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 1.f, 1e-6f);

  y->backward();
  float ln10 = static_cast<float>(std::numbers::ln10);
  EXPECT_NEAR(x->grad[0], 1.f / ln10, 1e-5f);
  EXPECT_NEAR(x->grad[1], 1.f / (10.f * ln10), 1e-5f);
}

/**
 * @brief Tests square root forward evaluation and 1/(2*sqrt(x)) gradient.
 */
TEST(grad_sqrt) {
  auto x = scorch::Tensor<float>::from({2}, {4.f, 9.f}, /*requires_grad=*/true);
  auto y = scorch::sqrt(x);
  EXPECT_NEAR(y->data[0], 2.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 3.f, 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 1.f / 4.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 1.f / 6.f, 1e-6f);
}

/**
 * @brief Tests reciprocal square root forward evaluation and -1/(2*x^(3/2)) gradient.
 */
TEST(grad_rsqrt) {
  auto x = scorch::Tensor<float>::from({2}, {4.f, 9.f}, /*requires_grad=*/true);
  auto y = scorch::rsqrt(x);
  EXPECT_NEAR(y->data[0], 0.5f, 1e-6f);
  EXPECT_NEAR(y->data[1], 1.f / 3.f, 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], -0.5f / 8.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], -0.5f / 27.f, 1e-6f);
}

/**
 * @brief Tests absolute value forward evaluation and sgn(x) gradient.
 */
TEST(grad_abs) {
  auto x = scorch::Tensor<float>::from({3}, {-3.f, 0.f, 4.f}, /*requires_grad=*/true);
  auto y = scorch::abs(x);
  EXPECT_NEAR(y->data[0], 3.f, 1e-6f);
  EXPECT_NEAR(y->data[1], 0.f, 1e-6f);
  EXPECT_NEAR(y->data[2], 4.f, 1e-6f);

  y->backward();
  EXPECT_NEAR(x->grad[0], -1.f, 1e-6f);
  EXPECT_NEAR(x->grad[1], 0.f, 1e-6f);
  EXPECT_NEAR(x->grad[2], 1.f, 1e-6f);
}
