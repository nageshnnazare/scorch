/**
 * @file main.cc
 * @brief Demonstration of scorch autograd, neural network modules (scorch::nn),
 *        and optimizers (scorch::optim).
 */

#include <iostream>

#include "scorch.h"

int main(int argc, char* argv[]) {
  std::cout << "=== Part 1: Low-Level Tensor & Autograd Pipeline ===" << std::endl;
  auto w = scorch::ones({2, 3}, /*requires_grad=*/true);
  auto b = scorch::full({2, 3}, 2.f, /*requires_grad=*/true);
  auto c = (w + b) * w;
  c->print("c = (w + b) * w");
  c->backward();
  std::cout << "w->grad[0] = " << w->grad[0] << " (expected 4)" << std::endl;
  std::cout << "b->grad[0] = " << b->grad[0] << " (expected 1)" << std::endl;

  std::cout << "\n=== Part 2: High-Level scorch::nn & scorch::optim ===" << std::endl;
  // Construct a 2-layer MLP: 2 -> 4 -> 1
  auto model = std::make_shared<scorch::nn::Sequential>(std::make_shared<scorch::nn::Linear>(2, 4),
                                                        std::make_shared<scorch::nn::ReLU>(),
                                                        std::make_shared<scorch::nn::Linear>(4, 1));

  std::cout << "Model Architecture:" << std::endl;
  for (const auto& [name, param] : model->named_parameters()) {
    std::cout << "  - Parameter: " << name << " shape=[";
    for (std::size_t i = 0; i < param->shape.size(); ++i) {
      std::cout << (i ? ", " : "") << param->shape[i];
    }
    std::cout << "]" << std::endl;
  }

  // Training data: XOR-like toy regression
  auto x = scorch::from({4, 2}, {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f});
  auto y = scorch::from({4, 1}, {0.f, 1.f, 1.f, 0.f});

  scorch::optim::Adam optimizer(model->parameters(), /*lr=*/0.05f);
  auto criterion = std::make_shared<scorch::nn::MSELoss>();

  std::cout << "\nTraining MLP with Adam..." << std::endl;
  for (int epoch = 1; epoch <= 100; ++epoch) {
    optimizer.zero_grad();
    auto pred = model->forward(x);
    auto loss = criterion->forward(pred, y);
    loss->backward();
    optimizer.step();

    if (epoch % 25 == 0 || epoch == 1) {
      std::cout << "  Epoch " << epoch << " | Loss: " << loss->data[0] << std::endl;
    }
  }

  std::cout << "\nFinal Predictions:" << std::endl;
  auto final_pred = model->forward(x);
  final_pred->print("model(x)");

  return 0;
}
