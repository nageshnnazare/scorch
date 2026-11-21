/**
 * @file test_ops_loss.cc
 * @brief Unit tests for loss functions: MSE Loss, Cross-Entropy Loss.
 */

#include "test_common.h"

TEST(grad_mse_loss) {
  auto pred = scorch::Tensor<float>::from({3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto target = scorch::Tensor<float>::from({3}, {1.5f, 2.5f, 3.5f}, /*requires_grad=*/false);
  auto loss = scorch::mse_loss(pred, target);
  EXPECT_NEAR(loss->data[0], 0.25f, 1e-5f);
  loss->backward();
  EXPECT_NEAR(pred->grad[0], -1.f / 3.f, 1e-5f);
  EXPECT_NEAR(pred->grad[1], -1.f / 3.f, 1e-5f);
  EXPECT_NEAR(pred->grad[2], -1.f / 3.f, 1e-5f);
}

TEST(grad_cross_entropy) {
  auto logits =
      scorch::Tensor<float>::from({2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f}, /*requires_grad=*/true);
  auto targets = scorch::Tensor<int>::from({2}, {2, 0});
  auto loss = scorch::cross_entropy_loss(logits, targets);
  EXPECT(loss->data[0] > 0.f);
  loss->backward();
  EXPECT(logits->grad.size() == 6);
  bool has_nonzero = false;
  for (auto g : logits->grad) {
    if (std::fabs(g) > 1e-7f) has_nonzero = true;
  }
  EXPECT(has_nonzero);
}
