#ifndef _SCORCH_OPTIM_SGD_H_
#define _SCORCH_OPTIM_SGD_H_

/**
 * @file sgd.h
 * @brief Stochastic Gradient Descent optimizer with momentum and weight decay (torch.optim.SGD).
 */

#include <vector>

#include "optim/optimizer.h"

namespace scorch::optim {

class SGD : public Optimizer {
 public:
  SGD(std::vector<std::shared_ptr<Tensor<float>>> parameters, float lr, float momentum = 0.0f,
      float weight_decay = 0.0f);

  void step() override;

 private:
  float lr_;
  float momentum_;
  float weight_decay_;
  std::vector<std::vector<float>> velocities_;
};

}  // namespace scorch::optim

#endif  // !_SCORCH_OPTIM_SGD_H_
