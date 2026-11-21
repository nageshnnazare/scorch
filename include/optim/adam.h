#ifndef _SCORCH_OPTIM_ADAM_H_
#define _SCORCH_OPTIM_ADAM_H_

/**
 * @file adam.h
 * @brief Adam optimizer with adaptive moment estimation (torch.optim.Adam).
 */

#include <vector>

#include "optim/optimizer.h"

namespace scorch::optim {

class Adam : public Optimizer {
 public:
  Adam(std::vector<std::shared_ptr<Tensor<float>>> parameters, float lr = 0.001f,
       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

  void step() override;

 private:
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  std::size_t step_count_{0};
  std::vector<std::vector<float>> m_;  // 1st moment estimate
  std::vector<std::vector<float>> v_;  // 2nd moment estimate
};

}  // namespace scorch::optim

#endif  // !_SCORCH_OPTIM_ADAM_H_
