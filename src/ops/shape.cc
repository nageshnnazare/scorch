/**
 * @file shape.cc
 * @brief Implementation of shape and view manipulations: reshape, flatten, squeeze, unsqueeze, cat.
 *
 * @details
 * Defines differentiable tensor transformations:
 * - `reshape`: View or change dimension extents preserving total element count.
 * - `flatten`: Flatten multidimensional tensor into a 1-D vector.
 * - `squeeze`: Remove a singleton dimension (size 1).
 * - `unsqueeze`: Insert a singleton dimension at specified axis.
 * - `cat`: Concatenate multiple tensors along an existing axis.
 */

#include "ops/shape.h"

#include <stdexcept>

#include "ops/ops_common.h"

namespace scorch {

/**
 * @brief Reshapes a tensor to a new shape of equal numel.
 * @param x Input tensor.
 * @param new_shape Target shape dimensions.
 * @return Tensor with updated shape.
 * @throws std::invalid_argument If numel(new_shape) != numel(x->shape).
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> reshape(const std::shared_ptr<Tensor<T>>& x, Shape new_shape) {
  if (numel(new_shape) != numel(x->shape)) {
    throw std::invalid_argument("reshape: numel mismatch");
  }

  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(std::move(new_shape), rg, "reshape", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = x->data[i];
  }

  if (rg) {
    out->backward_fn = [x]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i];
        }
      }
    };
  }

  return out;
}

/**
 * @brief Flattens a tensor into a 1-D tensor of shape {numel(x)}.
 * @param x Input tensor.
 * @return 1-D tensor containing all elements.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> flatten(const std::shared_ptr<Tensor<T>>& x) {
  return reshape(x, {numel(x->shape)});
}

/**
 * @brief Removes a singleton dimension of size 1 at the specified axis.
 * @param x Input tensor.
 * @param dim Index of dimension of size 1 to remove.
 * @return Tensor with dimension removed.
 * @throws std::invalid_argument If dim is out of bounds or shape[dim] != 1.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> squeeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim) {
  if (dim >= x->shape.size()) {
    throw std::invalid_argument("squeeze: dim out of range");
  }
  if (x->shape[dim] != 1) {
    throw std::invalid_argument("squeeze: dimension at dim is not size 1");
  }

  Shape new_shape;
  for (std::size_t i = 0; i < x->shape.size(); ++i) {
    if (i != dim) new_shape.push_back(x->shape[i]);
  }
  if (new_shape.empty()) new_shape.push_back(1);

  return reshape(x, std::move(new_shape));
}

/**
 * @brief Inserts a singleton dimension (size 1) at the specified axis.
 * @param x Input tensor.
 * @param dim Index where singleton dimension should be inserted (0 <= dim <= rank).
 * @return Tensor with dimension inserted.
 * @throws std::invalid_argument If dim > x->shape.size().
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> unsqueeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim) {
  if (dim > x->shape.size()) {
    throw std::invalid_argument("unsqueeze: dim out of range");
  }

  Shape new_shape = x->shape;
  new_shape.insert(new_shape.begin() + dim, 1);

  return reshape(x, std::move(new_shape));
}

/**
 * @brief Concatenates a sequence of tensors along a specified axis.
 * @param tensors List of tensors to concatenate.
 * @param dim Axis along which tensors are concatenated.
 * @return Concatenated tensor.
 * @throws std::invalid_argument If tensors is empty or shapes do not align.
 */
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cat(const std::vector<std::shared_ptr<Tensor<T>>>& tensors,
                               std::size_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: empty tensor list");
  }

  const auto& first_shape = tensors[0]->shape;
  if (dim >= first_shape.size()) {
    throw std::invalid_argument("cat: dim out of range");
  }

  std::size_t total_dim_size = 0;
  bool any_rg = false;
  for (const auto& t : tensors) {
    if (t->shape.size() != first_shape.size()) {
      throw std::invalid_argument("cat: rank mismatch among input tensors");
    }
    for (std::size_t d = 0; d < first_shape.size(); ++d) {
      if (d != dim && t->shape[d] != first_shape[d]) {
        throw std::invalid_argument("cat: dimension mismatch along non-cat axis");
      }
    }
    total_dim_size += t->shape[dim];
    if (t->requires_grad) any_rg = true;
  }

  Shape out_shape = first_shape;
  out_shape[dim] = total_dim_size;

  auto out = detail::make_result<T>(out_shape, any_rg, "cat", tensors);

  std::size_t outer = 1;
  for (std::size_t d = 0; d < dim; ++d) outer *= first_shape[d];
  std::size_t inner = 1;
  for (std::size_t d = dim + 1; d < first_shape.size(); ++d) inner *= first_shape[d];

  std::size_t out_dim_stride = inner;
  std::size_t out_outer_stride = total_dim_size * inner;

  for (std::size_t o = 0; o < outer; ++o) {
    std::size_t dim_offset = 0;
    for (const auto& t : tensors) {
      std::size_t cur_dim_size = t->shape[dim];
      for (std::size_t d = 0; d < cur_dim_size; ++d) {
        for (std::size_t i = 0; i < inner; ++i) {
          std::size_t src_idx = o * (cur_dim_size * inner) + d * inner + i;
          std::size_t dst_idx = o * out_outer_stride + (dim_offset + d) * out_dim_stride + i;
          out->data[dst_idx] = t->data[src_idx];
        }
      }
      dim_offset += cur_dim_size;
    }
  }

  if (any_rg) {
    out->backward_fn = [tensors, dim, outer, inner, total_dim_size, out_dim_stride,
                        out_outer_stride]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;

      for (const auto& t : tensors) {
        detail::ensure_grad(*t);
      }

      for (std::size_t o = 0; o < outer; ++o) {
        std::size_t dim_offset = 0;
        for (const auto& t : tensors) {
          std::size_t cur_dim_size = t->shape[dim];
          if (t->requires_grad) {
            for (std::size_t d = 0; d < cur_dim_size; ++d) {
              for (std::size_t i = 0; i < inner; ++i) {
                std::size_t src_idx = o * (cur_dim_size * inner) + d * inner + i;
                std::size_t dst_idx = o * out_outer_stride + (dim_offset + d) * out_dim_stride + i;
                t->grad[src_idx] += self->grad[dst_idx];
              }
            }
          }
          dim_offset += cur_dim_size;
        }
      }
    };
  }

  return out;
}

// Explicit template instantiations
template std::shared_ptr<Tensor<float>> reshape(const std::shared_ptr<Tensor<float>>&, Shape);
template std::shared_ptr<Tensor<double>> reshape(const std::shared_ptr<Tensor<double>>&, Shape);
template std::shared_ptr<Tensor<int>> reshape(const std::shared_ptr<Tensor<int>>&, Shape);

template std::shared_ptr<Tensor<float>> flatten(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> flatten(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> flatten(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> squeeze(const std::shared_ptr<Tensor<float>>&, std::size_t);
template std::shared_ptr<Tensor<double>> squeeze(const std::shared_ptr<Tensor<double>>&,
                                                 std::size_t);
template std::shared_ptr<Tensor<int>> squeeze(const std::shared_ptr<Tensor<int>>&, std::size_t);

template std::shared_ptr<Tensor<float>> unsqueeze(const std::shared_ptr<Tensor<float>>&,
                                                  std::size_t);
template std::shared_ptr<Tensor<double>> unsqueeze(const std::shared_ptr<Tensor<double>>&,
                                                   std::size_t);
template std::shared_ptr<Tensor<int>> unsqueeze(const std::shared_ptr<Tensor<int>>&, std::size_t);

template std::shared_ptr<Tensor<float>> cat(const std::vector<std::shared_ptr<Tensor<float>>>&,
                                            std::size_t);
template std::shared_ptr<Tensor<double>> cat(const std::vector<std::shared_ptr<Tensor<double>>>&,
                                             std::size_t);
template std::shared_ptr<Tensor<int>> cat(const std::vector<std::shared_ptr<Tensor<int>>>&,
                                          std::size_t);

}  // namespace scorch
