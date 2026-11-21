/**
 * @file main.cc
 * @brief Demonstration executable for scorch::Tensor construction, operators, and autograd.
 */

#include <iostream>

#include "scorch.h"

int main(int argc, char* argv[]) {
  // Construct a 2D weight matrix initialized to ones with gradient tracking enabled
  auto w = scorch::ones({2, 3}, /*requires_grad=*/true);
  w->print("w");

  // Construct another 2D matrix
  auto b = scorch::full({2, 3}, 2.f, /*requires_grad=*/true);
  b->print("b");

  // Demonstrate overloaded operator+ and operator*
  auto c = (w + b) * w;
  c->print("c = (w + b) * w");

  // Backpropagate
  c->backward();
  std::cout << "w->grad[0] = " << w->grad[0] << " (expected 4)" << std::endl;
  std::cout << "b->grad[0] = " << b->grad[0] << " (expected 1)" << std::endl;

  return 0;
}
