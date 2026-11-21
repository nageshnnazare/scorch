# scorch

A lightweight, PyTorch-equivalent C++20 tensor library and dynamic automatic differentiation (autograd) engine.

[![Standard](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key Features

- **Neural Networks (`scorch::nn`)**: PyTorch-like modular deep learning layer system:
  - `nn::Module`: Base class with parameter management (`parameters()`, `named_parameters()`), submodule hierarchy, and `train()`/`eval()` modes.
  - `nn::Linear`: Fully-connected layer with Kaiming uniform parameter initialization and rank-2 broadcasted bias.
  - `nn::Sequential`: Composability container for ordered multi-layer execution.
  - `nn::Dropout`: Inverted dropout supporting training vs. evaluation behavior.
  - Activations & Losses: `nn::ReLU`, `nn::Sigmoid`, `nn::Tanh`, `nn::LeakyReLU`, `nn::Softmax`, `nn::MSELoss`, `nn::CrossEntropyLoss`.
- **Optimizers (`scorch::optim`)**:
  - `optim::SGD`: Stochastic gradient descent with momentum and weight decay.
  - `optim::Adam`: Adaptive moment estimation with first/second order moments and bias correction.
- **Dynamic Reverse-Mode Autograd**: Builds a directed acyclic graph (DAG) during the forward pass and computes exact gradients via topological sort backpropagation (`tensor->backward()`).
- **PyTorch-Style API in C++**: Factory functions (`scorch::zeros`, `scorch::ones`, `scorch::full`, `scorch::rand`, `scorch::from`), functional operators, and activations mirror the PyTorch frontend.
- **Memory Safety & Graph Traversal**: Built on `std::shared_ptr<Tensor<T>>` and `std::enable_shared_from_this`, eliminating manual pointer lifetime hazards while capturing parent dependencies directly inside backward closures.
- **Natural Mathematical Syntax**: Overloaded operators (`+`, `-`, `*`, `/`, unary `-`, scalar operations) allow writing intuitive expression pipelines directly on `std::shared_ptr<Tensor<T>>`.
- **N-Dimensional Strided Tensors**: Supports arbitrary tensor shapes, strided layouts, slicing, views, and index access.
- **Modular Architecture**: Cleanly separated translation units under `include/ops/`, `include/nn/`, `include/optim/`, `src/`, and `tests/`.
- **Rich Operator Catalog**:
  - **Trigonometric & Hyperbolic**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`.
  - **Arithmetic & Advanced Math**: `add`, `sub`, `neg`, `mul`, `div`, `pow`, `exp`, `log`, `loge`, `log2`, `log10`, `sqrt`, `rsqrt`, `abs`, and scalar variants.
  - **Matrix & Linear Algebra**: `matmul`, `transpose`.
  - **Activations**: `relu`, `sigmoid`, `tanh`, `leaky_relu`, `softmax`.
  - **Reductions**: `reduce_sum`, `sum`, `mean`, `max`.
  - **Shape Manipulations**: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `cat`.
  - **Loss Functions**: `mse_loss`, `cross_entropy_loss`.

---

## Quickstart Examples

### 1. High-Level Neural Network Training (`scorch::nn` & `scorch::optim`)

```cpp
#include <iostream>
#include "scorch.h"

int main() {
  // Construct a 2-layer MLP (2 -> 4 -> 1)
  auto model = std::make_shared<scorch::nn::Sequential>(
      std::make_shared<scorch::nn::Linear>(2, 4),
      std::make_shared<scorch::nn::ReLU>(),
      std::make_shared<scorch::nn::Linear>(4, 1)
  );

  // Optimizer & Loss function
  scorch::optim::Adam optimizer(model->parameters(), /*lr=*/0.05f);
  auto criterion = std::make_shared<scorch::nn::MSELoss>();

  // Training data
  auto x = scorch::from({4, 2}, {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f});
  auto y = scorch::from({4, 1}, {0.f, 1.f, 1.f, 0.f});

  // Training loop
  for (int epoch = 1; epoch <= 100; ++epoch) {
    optimizer.zero_grad();
    auto pred = model->forward(x);
    auto loss = criterion->forward(pred, y);
    loss->backward();
    optimizer.step();
  }

  auto final_pred = model->forward(x);
  final_pred->print("model(x)");
  return 0;
}
```

### 2. Low-Level Tensor & Autograd Pipeline

```cpp
#include <iostream>
#include "scorch.h"

int main() {
  auto w = scorch::ones({2, 3}, /*requires_grad=*/true);
  auto b = scorch::full({2, 3}, 2.0f, /*requires_grad=*/true);

  // Dynamic computation graph
  auto c = (w + b) * w;
  c->backward();

  std::cout << "w->grad[0] = " << w->grad[0] << " (expected 4)\n";
  std::cout << "b->grad[0] = " << b->grad[0] << " (expected 1)\n";
  return 0;
}
```
---

## Building and Running

### Prerequisites

- **C++20 Compliant Compiler**: GCC 11+, Clang 13+, or Apple Clang 14+
- **CMake**: Version 3.20 or newer
- **Ninja**: (Recommended) or Make
- **clang-format**: Optional, for code formatting

### Build

```bash
# Configure build directory with Ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build the scorch library, demo binary, and test runner
ninja -C build
```

### Run the Demo

```bash
./build/scorch
```

### Run Tests

Each test case is registered individually with **CTest**:

```bash
# Run all tests with CTest
ctest --test-dir build --output-on-failure

# Alternatively, run via custom Ninja check target
ninja -C build check
```

### Code Formatting

Check or apply Google C++ style formatting using clang-format:

```bash
# In-place format all source files
ninja -C build format

# Verify formatting without modifying files (used in CI)
ninja -C build format-check
```

---

## License

This project is licensed under the [MIT License](LICENSE).
