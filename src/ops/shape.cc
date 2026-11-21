/**
 * @file shape.cc
 * @brief Implementation of shape and view manipulations: reshape, flatten, squeeze, unsqueeze, cat.
 */

#include "ops/shape.h"

#include <stdexcept>

#include "ops/ops_common.h"

namespace AG {

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

template <TensorScalar T>
std::shared_ptr<Tensor<T>> flatten(const std::shared_ptr<Tensor<T>>& x) {
  return reshape(x, {numel(x->shape)});
}

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

template <TensorScalar T>
std::shared_ptr<Tensor<T>> unsqueeze(const std::shared_ptr<Tensor<T>>& x, std::size_t dim) {
  if (dim > x->shape.size()) {
    throw std::invalid_argument("unsqueeze: dim out of range");
  }

  Shape new_shape = x->shape;
  new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(dim), 1);

  return reshape(x, std::move(new_shape));
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> cat(const std::vector<std::shared_ptr<Tensor<T>>>& tensors,
                               std::size_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: empty tensor list");
  }
  for (const auto& t : tensors) {
    if (!t) throw std::invalid_argument("cat: null tensor pointer");
  }
  if (tensors[0]->shape.size() != 2) {
    throw std::invalid_argument("cat: only rank-2 tensors are supported");
  }
  if (dim > 1) {
    throw std::invalid_argument("cat: dim must be 0 or 1 for rank-2 tensors");
  }

  const auto& ref_shape = tensors[0]->shape;
  for (std::size_t t = 1; t < tensors.size(); ++t) {
    if (tensors[t]->shape.size() != 2) {
      throw std::invalid_argument("cat: all tensors must be rank-2");
    }
    for (std::size_t d = 0; d < 2; ++d) {
      if (d != dim && tensors[t]->shape[d] != ref_shape[d]) {
        throw std::invalid_argument("cat: shape mismatch on non-cat dimension");
      }
    }
  }

  std::size_t cat_size = 0;
  for (const auto& t : tensors) cat_size += t->shape[dim];

  Shape out_shape = ref_shape;
  out_shape[dim] = cat_size;

  bool rg = false;
  for (const auto& t : tensors) {
    rg = rg || t->requires_grad;
  }

  auto out = detail::make_result<T>(out_shape, rg, "cat", tensors);

  if (dim == 0) {
    std::size_t offset = 0;
    for (const auto& t : tensors) {
      for (std::size_t i = 0; i < t->data.size(); ++i) {
        out->data[offset + i] = t->data[i];
      }
      offset += t->data.size();
    }
  } else {
    const std::size_t out_cols = out_shape[1];
    const std::size_t nrows = ref_shape[0];
    for (std::size_t r = 0; r < nrows; ++r) {
      std::size_t col_offset = 0;
      for (const auto& t : tensors) {
        for (std::size_t c = 0; c < t->shape[1]; ++c) {
          out->data[r * out_cols + col_offset + c] = t->data[r * t->shape[1] + c];
        }
        col_offset += t->shape[1];
      }
    }
  }

  if (rg) {
    std::vector<Shape> input_shapes;
    for (const auto& t : tensors) input_shapes.push_back(t->shape);

    out->backward_fn = [tensors, input_shapes, dim]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;

      for (auto& p : tensors) detail::ensure_grad(*p);

      if (dim == 0) {
        std::size_t offset = 0;
        for (std::size_t t = 0; t < tensors.size(); ++t) {
          if (tensors[t]->requires_grad) {
            std::size_t n = tensors[t]->data.size();
            for (std::size_t i = 0; i < n; ++i) {
              tensors[t]->grad[i] += self->grad[offset + i];
            }
          }
          offset += tensors[t]->data.size();
        }
      } else {
        const std::size_t nrows = input_shapes[0][0];
        const std::size_t out_cols = self->grad.size() / nrows;
        for (std::size_t r = 0; r < nrows; ++r) {
          std::size_t col_offset = 0;
          for (std::size_t t = 0; t < tensors.size(); ++t) {
            if (tensors[t]->requires_grad) {
              for (std::size_t c = 0; c < input_shapes[t][1]; ++c) {
                tensors[t]->grad[r * input_shapes[t][1] + c] +=
                    self->grad[r * out_cols + col_offset + c];
              }
            }
            col_offset += input_shapes[t][1];
          }
        }
      }
    };
  }

  return out;
}

// Explicit Instantiations
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

}  // namespace AG
