#ifndef _SCORCH_H_
#define _SCORCH_H_

/**
 * @file scorch.h
 * @brief Main umbrella header for the scorch library, mirroring PyTorch's `torch/torch.h`.
 *
 * @details
 * Includes the entire scorch deep learning ecosystem:
 * - `tensor.h`: Multidimensional tensor container and core autograd engine
 * - `tensor_ops.h`: Differentiable mathematical and linear algebra operations
 * - `nn/nn.h`: Neural network modules (Linear, Sequential, Conv/Dropout, Activations, Losses)
 * - `optim/optim.h`: First-order parameter optimizers (SGD, Adam)
 *
 * Example workflow:
 * @code{.cpp}
 * #include "scorch.h"
 *
 * int main() {
 *   // Define model
 *   auto model = std::make_shared<scorch::nn::Sequential>(
 *       std::make_shared<scorch::nn::Linear>(784, 128),
 *       std::make_shared<scorch::nn::ReLU>(),
 *       std::make_shared<scorch::nn::Linear>(128, 10));
 *
 *   scorch::optim::Adam optimizer(model->parameters(), 1e-3f);
 *   auto criterion = std::make_shared<scorch::nn::CrossEntropyLoss>();
 *   // ...
 * }
 * @endcode
 */

#include "nn/nn.h"
#include "optim/optim.h"
#include "tensor.h"
#include "tensor_ops.h"

#endif  // !_SCORCH_H_
