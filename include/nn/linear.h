#ifndef _SCORCH_NN_LINEAR_H_
#define _SCORCH_NN_LINEAR_H_

/**
 * @file linear.h
 * @brief Fully connected linear transformation layer (torch.nn.Linear).
 */

#include <cstddef>
#include <memory>

#include "nn/module.h"

namespace scorch::nn {

/**
 * @brief Applies an affine linear transformation to incoming data: y = x A^T + b or y = x W + b.
 */
class Linear : public Module {
 public:
  /**
   * @brief Construct a Linear layer.
   * @param in_features Size of each input sample.
   * @param out_features Size of each output sample.
   * @param bias If set to false, layer will not learn an additive bias. Default: true.
   */
  Linear(std::size_t in_features, std::size_t out_features, bool bias = true);

  std::shared_ptr<Tensor<float>> forward(const std::shared_ptr<Tensor<float>>& input) override;

  std::size_t in_features() const { return in_features_; }
  std::size_t out_features() const { return out_features_; }
  bool has_bias() const { return has_bias_; }

  std::shared_ptr<Tensor<float>> weight;
  std::shared_ptr<Tensor<float>> bias;

 private:
  std::size_t in_features_;
  std::size_t out_features_;
  bool has_bias_;
};

}  // namespace scorch::nn

#endif  // !_SCORCH_NN_LINEAR_H_
