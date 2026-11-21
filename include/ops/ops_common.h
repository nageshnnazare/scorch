#ifndef _SCORCH_OPS_COMMON_H_
#define _SCORCH_OPS_COMMON_H_

/**
 * @file ops_common.h
 * @brief Internal utility helpers shared across differentiable tensor operations.
 *
 * @details
 * Defines helper routines for gradient memory validation and DAG node construction
 * used by tensor operation kernels.
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"

namespace scorch {
namespace detail {

/**
 * @brief Ensures gradient storage vector is initialized to zero for a tensor requiring gradients.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param t Reference to the tensor whose gradient storage should be verified.
 */
template <TensorScalar T>
inline void ensure_grad(Tensor<T>& t) {
  if (t.requires_grad && t.grad.size() != t.data.size()) {
    t.grad.assign(t.data.size(), 0.f);
  }
}

/**
 * @brief Constructs a new result tensor pre-wired to parent nodes in the computation graph.
 *
 * @tparam T Underlying scalar type (must satisfy TensorScalar).
 * @param shape Output tensor dimensions.
 * @param requires_grad Flag indicating whether output tensor tracks autograd gradients.
 * @param op_name Diagnostic name of the operation (e.g. "Add", "Matmul") for debugging/profiling.
 * @param parents Vector of input operand tensors that produced this node in the DAG.
 * @return Newly allocated tensor wrapped in std::shared_ptr.
 */
template <TensorScalar T>
inline std::shared_ptr<Tensor<T>> make_result(Shape shape, bool requires_grad, std::string op_name,
                                              std::vector<std::shared_ptr<Tensor<T>>> parents) {
  auto out = Tensor<T>::zeros(std::move(shape), requires_grad);
  out->op_name = std::move(op_name);
  out->parents = std::move(parents);
  return out;
}

}  // namespace detail
}  // namespace scorch

#endif  // !_SCORCH_OPS_COMMON_H_
