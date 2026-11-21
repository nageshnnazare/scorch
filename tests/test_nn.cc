/**
 * @file test_nn.cc
 * @brief Unit tests for scorch::nn modules and scorch::optim optimizers.
 *
 * @details
 * Validates:
 * - Linear forward output shapes (2-D and 1-D inputs) and analytical gradients.
 * - Module parameter registration and recursive inspection.
 * - Sequential composition pipeline with activation modules.
 * - Dropout stochastic behavior vs deterministic eval mode.
 * - MSELoss and CrossEntropyLoss module wrappers.
 * - SGD and Adam optimization parameter step updates.
 * - End-to-end toy MLP neural network training convergence.
 */

#include "nn/nn.h"
#include "optim/optim.h"
#include "test_common.h"

// -----------------------------------------------------------------------------
// 1. Linear Layer Tests
// -----------------------------------------------------------------------------

/**
 * @brief Tests output shape of 2-D input to Linear layer.
 */
TEST(nn_linear_forward_shape) {
  auto fc = std::make_shared<scorch::nn::Linear>(4, 2, /*bias=*/true);
  auto x = scorch::ones({3, 4});
  auto y = fc->forward(x);

  EXPECT(y->shape.size() == 2);
  EXPECT(y->shape[0] == 3);
  EXPECT(y->shape[1] == 2);
}

/**
 * @brief Tests 1-D unbatched vector input support in Linear layer.
 */
TEST(nn_linear_1d_input) {
  auto fc = std::make_shared<scorch::nn::Linear>(3, 5, /*bias=*/true);
  auto x = scorch::ones({3});
  auto y = fc->forward(x);

  EXPECT(y->shape.size() == 1);
  EXPECT(y->shape[0] == 5);
}

/**
 * @brief Tests exact analytical gradient accumulation for Linear weights and bias.
 */
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

/**
 * @brief Tests recursive parameter and named_parameter discovery across nested modules.
 */
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

/**
 * @brief Tests sequential evaluation across multiple stacked layers and activations.
 */
TEST(nn_sequential_pipeline) {
  auto model = std::make_shared<scorch::nn::Sequential>(
      std::make_shared<scorch::nn::Linear>(2, 3), std::make_shared<scorch::nn::ReLU>(),
      std::make_shared<scorch::nn::Linear>(3, 1), std::make_shared<scorch::nn::Sigmoid>());

  auto x = scorch::ones({4, 2});
  auto y = model->forward(x);

  EXPECT(y->shape.size() == 2);
  EXPECT(y->shape[0] == 4);
  EXPECT(y->shape[1] == 1);
  for (auto val : y->data) {
    EXPECT(val >= 0.f && val <= 1.f);
  }
}

/**
 * @brief Tests Dropout evaluation mode pass-through behavior.
 */
TEST(nn_dropout_eval_mode) {
  auto drop = std::make_shared<scorch::nn::Dropout>(0.5f);
  auto x = scorch::ones({4, 4});

  drop->eval();
  EXPECT(!drop->is_training());

  auto y = drop->forward(x);
  for (std::size_t i = 0; i < x->data.size(); ++i) {
    EXPECT_NEAR(y->data[i], x->data[i], 1e-6f);
  }
}

// -----------------------------------------------------------------------------
// 4. Loss Modules
// -----------------------------------------------------------------------------

/**
 * @brief Tests MSELoss and CrossEntropyLoss module wrappers.
 */
TEST(nn_loss_modules) {
  auto mse = std::make_shared<scorch::nn::MSELoss>();
  auto p = scorch::ones({2, 2});
  auto t = scorch::zeros({2, 2});
  auto l_mse = mse->forward(p, t);
  EXPECT_NEAR(l_mse->data[0], 1.f, 1e-6f);

  auto ce = std::make_shared<scorch::nn::CrossEntropyLoss>();
  auto logits = scorch::ones({2, 3});
  auto targets = scorch::Tensor<int>::from({2}, {0, 1});
  auto l_ce = ce->forward(logits, targets);
  EXPECT(l_ce->data[0] > 0.f);
}

// -----------------------------------------------------------------------------
// 5. Optimizers
// -----------------------------------------------------------------------------

/**
 * @brief Tests parameter updates performed by SGD with momentum and weight decay.
 */
TEST(optim_sgd_step) {
  auto w = scorch::ones({2}, /*requires_grad=*/true);
  w->grad = {0.1f, 0.2f};

  scorch::optim::SGD opt({w}, /*lr=*/0.1f, /*momentum=*/0.9f, /*weight_decay=*/0.01f);
  opt.step();

  EXPECT(w->data[0] < 1.0f);
  EXPECT(w->data[1] < 1.0f);
}

/**
 * @brief Tests parameter updates performed by Adam optimizer.
 */
TEST(optim_adam_step) {
  auto w = scorch::ones({2}, /*requires_grad=*/true);
  w->grad = {0.1f, 0.2f};

  scorch::optim::Adam opt({w}, /*lr=*/1e-3f);
  opt.step();

  EXPECT(w->data[0] < 1.0f);
  EXPECT(w->data[1] < 1.0f);
}

// -----------------------------------------------------------------------------
// 6. End-to-End Training Loop Verification
// -----------------------------------------------------------------------------

/**
 * @brief Tests end-to-end multi-epoch regression training convergence with Adam.
 */
TEST(nn_end_to_end_training) {
  auto model = std::make_shared<scorch::nn::Sequential>(std::make_shared<scorch::nn::Linear>(2, 4),
                                                        std::make_shared<scorch::nn::Tanh>(),
                                                        std::make_shared<scorch::nn::Linear>(4, 1));

  auto criterion = std::make_shared<scorch::nn::MSELoss>();
  scorch::optim::Adam optimizer(model->parameters(), /*lr=*/0.05f);

  auto x = scorch::from({4, 2}, {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f});
  auto y = scorch::from({4, 1}, {0.f, 1.f, 1.f, 0.f});

  float initial_loss = 0.f;
  float final_loss = 0.f;

  for (int epoch = 0; epoch < 50; ++epoch) {
    optimizer.zero_grad();
    auto pred = model->forward(x);
    auto loss = criterion->forward(pred, y);
    if (epoch == 0) initial_loss = loss->data[0];
    if (epoch == 49) final_loss = loss->data[0];

    loss->backward();
    optimizer.step();
  }

  EXPECT(final_loss < initial_loss);
}
