#ifndef _SCORCH_OPTIM_OPTIM_H_
#define _SCORCH_OPTIM_OPTIM_H_

/**
 * @file optim.h
 * @brief Umbrella header for the `scorch::optim` optimization library.
 *
 * @details
 * Includes all parameter optimizers implemented in scorch, equivalent to PyTorch's
 * `torch.optim` module.
 *
 * Available optimizers:
 * - `scorch::optim::Optimizer`: Base class interface.
 * - `scorch::optim::SGD`: Stochastic Gradient Descent with momentum and weight decay.
 * - `scorch::optim::Adam`: Adaptive Moment Estimation with bias correction.
 *
 * @code{.cpp}
 * #include "optim/optim.h"
 *
 * scorch::optim::Adam optimizer(model->parameters(), 1e-3f);
 * @endcode
 *
 * @defgroup optim Optimizers
 * @brief First-order optimization algorithms for training neural network parameters.
 */

#include "optim/adam.h"
#include "optim/optimizer.h"
#include "optim/sgd.h"

#endif  // !_SCORCH_OPTIM_OPTIM_H_
