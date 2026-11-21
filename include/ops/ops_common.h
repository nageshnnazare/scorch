#ifndef _SCORCH_OPS_COMMON_H_
#define _SCORCH_OPS_COMMON_H_

/**
 * @file ops_common.h
 * @brief Internal helper functions shared across tensor operation implementations.
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"

namespace scorch {
namespace detail {

/**
 * @brief Ensures gradient storage is allocated for a tensor that requires gradients.
 */
template <TensorScalar T>
inline void ensure_grad(Tensor<T>& t) {
  if (t.requires_grad && t.grad.size() != t.data.size()) {
    t.grad.assign(t.data.size(), 0.f);
  }
}

/**
 * @brief Constructs an operation result tensor linked to its computation graph parents.
 */
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> make_result(Shape shape, bool requires_grad, std::string op_name,
                                              std::vector<std::shared_ptr<Tensor<T>>> parents) {
  auto out = Tensor<T>::zeros(std::move(shape), /*requires_grad=*/requires_grad);
  out->op_name = std::move(op_name);
  out->parents = std::move(parents);
  return out;
}

}  // namespace detail
}  // namespace scorch

#endif  // !_SCORCH_OPS_COMMON_H_
