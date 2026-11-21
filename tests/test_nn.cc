/**
 * @file test_nn.cc
 * @brief Unit tests for scorch::nn modules and scorch::optim optimizers.
 */

#include "nn/nn.h"
#include "optim/optim.h"
#include "test_common.h"

// -----------------------------------------------------------------------------
// 1. Linear Layer Tests
// -----------------------------------------------------------------------------

TEST(nn_linear_forward_shape) {
  auto fc = std::make_shared<scorch::nn::Linear>(4, 2, /*bias=*/true);
  auto x = scorch::ones({3, 4});
  auto y = fc->forward(x);

  EXPECT(y->shape.size() == 2);
  EXPECT(y->shape[0] == 3);
  EXPECT(y->shape[1] == 2);
}

TEST(nn_linear_1d_input) {
  auto fc = std::make_shared<scorch::nn::Linear>(3, 5, /*bias=*/true);
  auto x = scorch::ones({3});
  auto y = fc->forward(x);

  EXPECT(y->shape.size() == 1);
  EXPECT(y->shape[0] == 5);
}

TEST(nn_linear_gradients) {
  auto fc = std::make_shared<scorch::nn::Linear>(2, 2, /*bias=*/true);
  // Explicitly set weights and bias for deterministic testing
  fc->weight->data = {1.f, 2.f, 3.f, 4.f};
  fc->bias->data = {0.5f, -0.5f};

  auto x = scorch::from({1, 2}, {1.f, 1.f}, /*requires_grad=*/true);
  auto y = fc->forward(x);

  // y = [1*1 + 1*3 + 0.5, 1*2 + 1*4 - 0.5] = [4.5, 5.5]
  EXPECT_NEAR(y->data[0], 4.5f, 1e-4f);
  EXPECT_NEAR(y->data[1], 5.5f, 1e-4f);

  auto loss = scorch::reduce(y);  // 4.5 + 5.5 = 10
  loss->backward();

  EXPECT(fc->weight->grad.size() == 4);
  EXPECT(fc->bias->grad.size() == 2);
  // dy/dw_ij = x_i = 1.f
  EXPECT_NEAR(fc->weight->grad[0], 1.f, 1e-4f);
  EXPECT_NEAR(fc->weight->grad[1], 1.f, 1e-4f);
  EXPECT_NEAR(fc->weight->grad[2], 1.f, 1e-4f);
  EXPECT_NEAR(fc->weight->grad[3], 1.f, 1e-4f);
  // dy/db_j = 1.f
  EXPECT_NEAR(fc->bias->grad[0], 1.f, 1e-4f);
  EXPECT_NEAR(fc->bias->grad[1], 1.f, 1e-4f);
}

// -----------------------------------------------------------------------------
// 2. Module Hierarchy & Parameters
// -----------------------------------------------------------------------------

TEST(nn_module_parameters) {
  auto model = std::make_shared<scorch::nn::Sequential>(
      std::make_shared<scorch::nn::Linear>(3, 4, /*bias=*/true),
      std::make_shared<scorch::nn::ReLU>(),
      std::make_shared<scorch::nn::Linear>(4, 1, /*bias=*/false));

  auto params = model->parameters();
  // Layer 1: weight (12) + bias (4)
  // Layer 2: ReLU (0)
  // Layer 3: weight (4)
  // Total parameters count = 3 tensors
  EXPECT(params.size() == 3);

  auto named_params = model->named_parameters();
  EXPECT(named_params.size() == 3);
  EXPECT(named_params[0].first == "0.weight");
  EXPECT(named_params[1].first == "0.bias");
  EXPECT(named_params[2].first == "2.weight");

  model->zero_grad();
}

// -----------------------------------------------------------------------------
// 3. Activations and Sequential
// -----------------------------------------------------------------------------

TEST(nn_sequential_pipeline) {
  auto model = std::make_shared<scorch::nn::Sequential>(
      std::make_shared<scorch::nn::Linear>(2, 3), std::make_shared<scorch::nn::ReLU>(),
      std::make_shared<scorch::nn::Linear>(3, 1), std::make_shared<scorch::nn::Sigmoid>());

  auto x = scorch::ones({4, 2});
  auto y = model->forward(x);

  EXPECT(y->shape.size() == 2);
  EXPECT(y->shape[0] == 4);
  EXPECT(y->shape[1] == 1);
  // Sigmoid outputs must be strictly in (0, 1)
  for (float val : y->data) {
    EXPECT(val > 0.0f && val < 1.0f);
  }
}

TEST(nn_dropout_eval_mode) {
  auto drop = std::make_shared<scorch::nn::Dropout>(0.5f);
  auto x = scorch::ones({4, 4});

  drop->eval();
  auto y_eval = drop->forward(x);
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_NEAR(y_eval->data[i], 1.0f, 1e-5f);
  }
}

// -----------------------------------------------------------------------------
// 4. Loss Modules
// -----------------------------------------------------------------------------

TEST(nn_loss_modules) {
  auto mse = std::make_shared<scorch::nn::MSELoss>();
  auto pred = scorch::from({2}, {1.0f, 2.0f}, /*requires_grad=*/true);
  auto target = scorch::from({2}, {1.0f, 4.0f});

  auto loss = mse->forward(pred, target);
  // ((1-1)^2 + (2-4)^2) / 2 = 4 / 2 = 2.0
  EXPECT_NEAR(loss->data[0], 2.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// 5. Optimizers (SGD & Adam)
// -----------------------------------------------------------------------------

TEST(optim_sgd_step) {
  auto p = scorch::from({2}, {5.0f, 10.0f}, /*requires_grad=*/true);
  scorch::optim::SGD opt({p}, /*lr=*/0.1f, /*momentum=*/0.9f);

  // Simulate loss = p[0] + p[1] -> grad = [1, 1]
  p->grad = {1.0f, 1.0f};
  opt.step();

  // v = 0.9 * 0 + 1 = 1.0; p = p - 0.1 * 1.0 = [4.9, 9.9]
  EXPECT_NEAR(p->data[0], 4.9f, 1e-4f);
  EXPECT_NEAR(p->data[1], 9.9f, 1e-4f);
}

TEST(optim_adam_step) {
  auto p = scorch::from({2}, {1.0f, 2.0f}, /*requires_grad=*/true);
  scorch::optim::Adam opt({p}, /*lr=*/0.01f);

  p->grad = {0.5f, -0.5f};
  opt.step();

  // Parameter should decrease where grad > 0, increase where grad < 0
  EXPECT(p->data[0] < 1.0f);
  EXPECT(p->data[1] > 2.0f);
}

TEST(nn_end_to_end_training) {
  // Fit a single linear layer y = 2x + 1
  auto model = std::make_shared<scorch::nn::Linear>(1, 1, /*bias=*/true);
  model->weight->data = {0.0f};
  model->bias->data = {0.0f};

  scorch::optim::SGD optimizer(model->parameters(), /*lr=*/0.05f);
  auto criterion = std::make_shared<scorch::nn::MSELoss>();

  auto x = scorch::from({2, 1}, {1.0f, 2.0f});
  auto y_true = scorch::from({2, 1}, {3.0f, 5.0f});

  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  for (int epoch = 0; epoch < 50; ++epoch) {
    optimizer.zero_grad();
    auto y_pred = model->forward(x);
    auto loss = criterion->forward(y_pred, y_true);

    if (epoch == 0) initial_loss = loss->data[0];
    if (epoch == 49) final_loss = loss->data[0];

    loss->backward();
    optimizer.step();
  }

  // Loss must decrease significantly
  EXPECT(final_loss < initial_loss);
  EXPECT(final_loss < 0.1f);
}
