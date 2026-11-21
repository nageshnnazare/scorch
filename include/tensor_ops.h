#ifndef _SCORCH_TENSOR_OPS_H_
#define _SCORCH_TENSOR_OPS_H_

/**
 * @file tensor_ops.h
 * @brief Umbrella header providing all differentiable tensor operations.
 *
 * @details
 * Including this single header imports all autograd-aware operations:
 * - `ops/activation.h`: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
 * - `ops/arithmetic.h`: Addition, subtraction, multiplication, division, power, exp, log
 * - `ops/loss.h`: MSE Loss, Cross-Entropy Loss
 * - `ops/matrix.h`: Matrix multiplication (matmul) and transposition
 * - `ops/reduction.h`: Sum, mean, max, sum along dimension
 * - `ops/shape.h`: Reshape, flatten, squeeze, unsqueeze, concatenate (cat)
 * - `ops/trigonometric.h`: Trigonometric, hyperbolic, log bases, sqrt, rsqrt, abs
 *
 * @defgroup ops Tensor Operations
 * @brief Differentiable element-wise, linear algebra, shape manipulation, and reduction operations.
 */

#include "ops/activation.h"
#include "ops/arithmetic.h"
#include "ops/loss.h"
#include "ops/matrix.h"
#include "ops/reduction.h"
#include "ops/shape.h"
#include "ops/trigonometric.h"

#endif  // !_SCORCH_TENSOR_OPS_H_
