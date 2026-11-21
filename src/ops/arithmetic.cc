/**
 * @file arithmetic.cc
 * @brief Implementation of element-wise arithmetic and scalar operations.
 */

#include "ops/arithmetic.h"

#include <cmath>
#include <stdexcept>

#include "ops/ops_common.h"

namespace AG {

template <TensorScalar T>
std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b) {
  if (a->shape != b->shape) {
    throw std::invalid_argument("add: shape mismatch");
  }

  bool rg = a->requires_grad || b->requires_grad;
  auto out = detail::make_result<T>(a->shape, rg, "add", {a, b});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }

  if (rg) {
    out->backward_fn = [a, b]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*a);
      detail::ensure_grad(*b);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        if (a->requires_grad) {
          a->grad[i] += self->grad[i];
        }
        if (b->requires_grad) {
          b->grad[i] += self->grad[i];
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> sub(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b) {
  if (a->shape != b->shape) {
    throw std::invalid_argument("sub: shape mismatch");
  }

  bool rg = a->requires_grad || b->requires_grad;
  auto out = detail::make_result<T>(a->shape, rg, "sub", {a, b});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }

  if (rg) {
    out->backward_fn = [a, b]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*a);
      detail::ensure_grad(*b);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        if (a->requires_grad) {
          a->grad[i] += self->grad[i];
        }
        if (b->requires_grad) {
          b->grad[i] += -self->grad[i];
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b) {
  if (a->shape != b->shape) {
    throw std::invalid_argument("mul: shape mismatch");
  }

  bool rg = a->requires_grad || b->requires_grad;
  auto out = detail::make_result<T>(a->shape, rg, "mul", {a, b});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = a->data[i] * b->data[i];
  }

  if (rg) {
    out->backward_fn = [a, b]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*a);
      detail::ensure_grad(*b);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        if (a->requires_grad) {
          a->grad[i] += self->grad[i] * static_cast<float>(b->data[i]);
        }
        if (b->requires_grad) {
          b->grad[i] += self->grad[i] * static_cast<float>(a->data[i]);
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> div(const std::shared_ptr<Tensor<T>>& a,
                               const std::shared_ptr<Tensor<T>>& b) {
  if (a->shape != b->shape) {
    throw std::invalid_argument("div: shape mismatch");
  }

  bool rg = a->requires_grad || b->requires_grad;
  auto out = detail::make_result<T>(a->shape, rg, "div", {a, b});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = a->data[i] / b->data[i];
  }

  if (rg) {
    out->backward_fn = [a, b]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*a);
      detail::ensure_grad(*b);

      for (std::size_t i = 0; i < self->grad.size(); ++i) {
        float bv = static_cast<float>(b->data[i]);
        if (a->requires_grad) {
          a->grad[i] += self->grad[i] / bv;
        }
        if (b->requires_grad) {
          float av = static_cast<float>(a->data[i]);
          b->grad[i] += -self->grad[i] * av / (bv * bv);
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> neg(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "neg", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = -x->data[i];
  }

  if (rg) {
    out->backward_fn = [x]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += -self->grad[i];
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> pow(const std::shared_ptr<Tensor<T>>& x, T exponent) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "pow", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] =
        static_cast<T>(std::pow(static_cast<double>(x->data[i]), static_cast<double>(exponent)));
  }

  if (rg) {
    out->backward_fn = [x, exponent]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float xv = static_cast<float>(x->data[i]);
          float exp_f = static_cast<float>(exponent);
          x->grad[i] += self->grad[i] * exp_f * static_cast<float>(std::pow(xv, exp_f - 1.f));
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> exp(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "exp", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::exp(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i] * static_cast<float>(out_data[i]);
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "log", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::log(static_cast<double>(x->data[i])));
  }

  if (rg) {
    out->backward_fn = [x]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i] / static_cast<float>(x->data[i]);
        }
      }
    };
  }

  return out;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> add_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "add_scalar", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = x->data[i] + scalar;
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
std::shared_ptr<Tensor<T>> mul_scalar(const std::shared_ptr<Tensor<T>>& x, T scalar) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "mul_scalar", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = x->data[i] * scalar;
  }

  if (rg) {
    out->backward_fn = [x, scalar]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        float s = static_cast<float>(scalar);
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i] * s;
        }
      }
    };
  }

  return out;
}

// Explicit Instantiations
template std::shared_ptr<Tensor<float>> add(const std::shared_ptr<Tensor<float>>&,
                                            const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> add(const std::shared_ptr<Tensor<double>>&,
                                             const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> add(const std::shared_ptr<Tensor<int>>&,
                                          const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> sub(const std::shared_ptr<Tensor<float>>&,
                                            const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> sub(const std::shared_ptr<Tensor<double>>&,
                                             const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> sub(const std::shared_ptr<Tensor<int>>&,
                                          const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> mul(const std::shared_ptr<Tensor<float>>&,
                                            const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> mul(const std::shared_ptr<Tensor<double>>&,
                                             const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> mul(const std::shared_ptr<Tensor<int>>&,
                                          const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> div(const std::shared_ptr<Tensor<float>>&,
                                            const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> div(const std::shared_ptr<Tensor<double>>&,
                                             const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> div(const std::shared_ptr<Tensor<int>>&,
                                          const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> neg(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> neg(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> neg(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> pow(const std::shared_ptr<Tensor<float>>&, float);
template std::shared_ptr<Tensor<double>> pow(const std::shared_ptr<Tensor<double>>&, double);
template std::shared_ptr<Tensor<int>> pow(const std::shared_ptr<Tensor<int>>&, int);

template std::shared_ptr<Tensor<float>> exp(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> exp(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> exp(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> log(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> log(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> log(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> add_scalar(const std::shared_ptr<Tensor<float>>&, float);
template std::shared_ptr<Tensor<double>> add_scalar(const std::shared_ptr<Tensor<double>>&, double);
template std::shared_ptr<Tensor<int>> add_scalar(const std::shared_ptr<Tensor<int>>&, int);

template std::shared_ptr<Tensor<float>> mul_scalar(const std::shared_ptr<Tensor<float>>&, float);
template std::shared_ptr<Tensor<double>> mul_scalar(const std::shared_ptr<Tensor<double>>&, double);
template std::shared_ptr<Tensor<int>> mul_scalar(const std::shared_ptr<Tensor<int>>&, int);

}  // namespace AG
