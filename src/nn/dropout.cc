/**
 * @file dropout.cc
 * @brief Implementation of the Dropout regularisation module.
 *
 * @details
 * ### Algorithm — Inverted Dropout
 *
 * At each call during training a fresh Bernoulli mask is sampled:
 *
 * ```
 * mask[i] = Bernoulli(1 - p) ? scale : 0,   scale = 1 / (1 - p)
 * output  = input * mask                     (element-wise)
 * ```
 *
 * Scaling the survivors at training time by \f$\frac{1}{1-p}\f$ preserves
 * the expected value of each activation, so **no rescaling is needed at
 * inference time**.
 *
 * ### Random number generation
 * Uses a function-static `std::mt19937` seeded once from `std::random_device`.
 * The generator is shared across all calls on the same thread; it is **not**
 * thread-safe.  For multi-threaded training, use separate `Dropout` instances
 * per thread or add external synchronisation.
 *
 * ### Autograd
 * The mask tensor is created with `requires_grad = false`.  The element-wise
 * multiply `input * mask` records the product in the autograd graph; during
 * the backward pass, gradients flow only through indices where `mask[i] != 0`.
 */

#include "nn/dropout.h"

#include <random>
#include <stdexcept>

#include "ops/arithmetic.h"

namespace scorch::nn {

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

Dropout::Dropout(float p) : p_(p) {
  // Validate that p is a legal probability for a drop rate.
  // p == 1 would zero every element, making the layer degenerate.
  if (p_ < 0.0f || p_ >= 1.0f) {
    throw std::invalid_argument("Dropout: probability p must satisfy 0 <= p < 1, got " +
                                std::to_string(p_));
  }
}

// -----------------------------------------------------------------------------
// Forward pass
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor<float>> Dropout::forward(const std::shared_ptr<Tensor<float>>& input) {
  // ── Eval mode / p == 0: no-op ─────────────────────────────────────────────
  // In evaluation mode dropout is disabled; return the input unchanged.
  // Also short-circuit when p == 0 (keep all, scale = 1).
  if (!is_training() || p_ <= 0.0f) {
    return input;
  }

  // ── Build the inverted-dropout mask ───────────────────────────────────────
  // scale = 1 / (1 - p): surviving activations are amplified so that the
  // expected sum of outputs equals the expected sum at test time.
  float scale = 1.0f / (1.0f - p_);

  // A Bernoulli draw with keep-probability (1 - p_).
  // Static generator: seeded once at first call, reused thereafter.
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::bernoulli_distribution dist(1.0f - p_);

  std::vector<float> mask_data(input->data.size());
  for (std::size_t i = 0; i < mask_data.size(); ++i) {
    // Keep → scale; drop → 0.
    mask_data[i] = dist(gen) ? scale : 0.0f;
  }

  // ── Apply mask ────────────────────────────────────────────────────────────
  // The mask has requires_grad = false; only 'input' carries the gradient.
  // The backward pass for element-wise multiply passes grad through non-zero
  // mask entries multiplied by 'scale'.
  auto mask = scorch::from(input->shape, std::move(mask_data), /*requires_grad=*/false);
  return input * mask;
}

}  // namespace scorch::nn
