#ifndef _SCORCH_NN_NN_H_
#define _SCORCH_NN_NN_H_

/**
 * @file nn.h
 * @brief Umbrella header for the `scorch::nn` neural network library.
 *
 * @details
 * Include this single header to get access to all built-in modules:
 *
 * ```cpp
 * #include "nn/nn.h"
 * ```
 *
 * ### Modules available
 *
 * | Header          | Contents                                              |
 * |-----------------|-------------------------------------------------------|
 * | `module.h`      | `Module` — abstract base class for all layers         |
 * | `linear.h`      | `Linear` — fully-connected affine layer               |
 * | `activations.h` | `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softmax`    |
 * | `sequential.h`  | `Sequential` — ordered module container               |
 * | `dropout.h`     | `Dropout` — stochastic regularisation                 |
 * | `loss.h`        | `MSELoss`, `CrossEntropyLoss`                         |
 *
 * ### Quick-start example
 * @code{.cpp}
 * #include "nn/nn.h"
 * #include "optim/optim.h"
 *
 * // Build a 2-layer MLP for 10-class classification
 * auto model = std::make_shared<scorch::nn::Sequential>(
 *     std::make_shared<scorch::nn::Linear>(784, 256),
 *     std::make_shared<scorch::nn::ReLU>(),
 *     std::make_shared<scorch::nn::Dropout>(0.3f),
 *     std::make_shared<scorch::nn::Linear>(256, 10));
 *
 * auto criterion = std::make_shared<scorch::nn::CrossEntropyLoss>();
 * scorch::optim::Adam optimizer(model->parameters(), 1e-3f);
 *
 * // Training step
 * model->train();
 * model->zero_grad();
 * auto logits = model->forward(x_batch);
 * auto loss   = criterion->forward(logits, y_batch);
 * loss->backward();
 * optimizer.step();
 * @endcode
 *
 * @ingroup nn
 */

#include "nn/activations.h"
#include "nn/dropout.h"
#include "nn/linear.h"
#include "nn/loss.h"
#include "nn/module.h"
#include "nn/sequential.h"

#endif  // !_SCORCH_NN_NN_H_
