#ifndef _SCORCH_NN_DROPOUT_H_
#define _SCORCH_NN_DROPOUT_H_

/**
 * @file dropout.h
 * @brief During training, randomly zeroes some of the elements with probability p
 * (torch.nn.Dropout).
 */

#include <memory>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Applies Dropout during training mode; acts as identity in eval mode.
 */
class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f);

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  float p() const { return p_; }

 private:
  float p_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_DROPOUT_H_
