# scorch

A lightweight, PyTorch-equivalent C++20 tensor library and dynamic automatic differentiation (autograd) engine.

[![Standard](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key Features

- **Dynamic Reverse-Mode Autograd**: Builds a directed acyclic graph (DAG) during the forward pass and computes exact gradients via topological sort backpropagation (`tensor->backward()`).
- **PyTorch-Style API in C++**: Factory functions (`scorch::zeros`, `scorch::ones`, `scorch::full`, `scorch::rand`, `scorch::from`), functional operators, and activations mirror the PyTorch frontend.
- **Memory Safety & Graph Traversal**: Built on `std::shared_ptr<Tensor<T>>` and `std::enable_shared_from_this`, eliminating manual pointer lifetime hazards while capturing parent dependencies directly inside backward closures.
- **Natural Mathematical Syntax**: Overloaded operators (`+`, `-`, `*`, `/`, unary `-`, scalar operations) allow writing intuitive expression pipelines directly on `std::shared_ptr<Tensor<T>>`.
- **N-Dimensional Strided Tensors**: Supports arbitrary tensor shapes, strided layouts, slicing, views, and index access.
- **Rich Operator Library**:
  - **Arithmetic**: `add`, `sub`, `neg`, `mul`, `div`, `pow`, `exp`, `log`, and scalar arithmetic.
  - **Matrix & Linear Algebra**: `matmul`, `transpose`.
  - **Activations**: `relu`, `sigmoid`, `tanh`, `leaky_relu`, `softmax`.
  - **Reductions**: `reduce_sum`, `sum`, `mean`, `max`.
  - **Shape Manipulations**: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `cat`.
  - **Loss Functions**: `mse_loss`, `cross_entropy_loss`.
- **Modular Architecture**: Cleanly separated translation units under `include/ops/`, `src/ops/`, and `tests/`.

---

## Quickstart Example

```cpp
#include <iostream>
#include "scorch.h"

int main() {
  // 1. Create tensors with gradient tracking (PyTorch-style free factories)
  auto w = scorch::ones({2, 3}, /*requires_grad=*/true);
  auto b = scorch::full({2, 3}, 2.0f, /*requires_grad=*/true);

  // 2. Build computation graph using natural overloaded operators
  auto c = (w + b) * w;

  // 3. Compute gradients via reverse-mode automatic differentiation
  c->backward();

  // 4. Access computed gradients
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
