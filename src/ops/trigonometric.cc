/**
 * @file trigonometric.cc
 * @brief Implementation of differentiable trigonometric and mathematical operations.
 */

#include "ops/trigonometric.h"

#include <cmath>
#include <numbers>

#include "ops/ops_common.h"

namespace scorch {

// -----------------------------------------------------------------------------
// sin(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sin(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "sin", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::sin(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float cos_val = static_cast<float>(std::cos(static_cast<double>(x_data[i])));
          x->grad[i] += self->grad[i] * cos_val;
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// cos(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cos(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "cos", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::cos(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float sin_val = static_cast<float>(std::sin(static_cast<double>(x_data[i])));
          x->grad[i] += self->grad[i] * (-sin_val);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// tan(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> tan(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "tan", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::tan(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float tan_val = static_cast<float>(out_data[i]);
          x->grad[i] += self->grad[i] * (1.0f + tan_val * tan_val);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// asin(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> asin(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "asin", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::asin(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float xv = static_cast<float>(x_data[i]);
          float denom = std::sqrt(1.0f - xv * xv);
          x->grad[i] += self->grad[i] / denom;
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// acos(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> acos(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "acos", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::acos(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float xv = static_cast<float>(x_data[i]);
          float denom = std::sqrt(1.0f - xv * xv);
          x->grad[i] += self->grad[i] * (-1.0f / denom);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// atan(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> atan(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "atan", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::atan(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float xv = static_cast<float>(x_data[i]);
          x->grad[i] += self->grad[i] / (1.0f + xv * xv);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// sinh(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sinh(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "sinh", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::sinh(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float cosh_val = static_cast<float>(std::cosh(static_cast<double>(x_data[i])));
          x->grad[i] += self->grad[i] * cosh_val;
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// cosh(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> cosh(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "cosh", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::cosh(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float sinh_val = static_cast<float>(std::sinh(static_cast<double>(x_data[i])));
          x->grad[i] += self->grad[i] * sinh_val;
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// log2(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log2(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "log2", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::log2(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      constexpr float ln2 = 0.69314718056f;
      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i] / (static_cast<float>(x_data[i]) * ln2);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// log10(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> log10(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "log10", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::log10(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      constexpr float ln10 = 2.30258509299f;
      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i] / (static_cast<float>(x_data[i]) * ln10);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// sqrt(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> sqrt(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "sqrt", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::sqrt(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          x->grad[i] += self->grad[i] / (2.0f * static_cast<float>(out_data[i]));
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// rsqrt(x): 1 / sqrt(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> rsqrt(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "rsqrt", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(1.0 / std::sqrt(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto out_data = out->data;
    out->backward_fn = [x, out_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float y = static_cast<float>(out_data[i]);
          // d/dx(x^{-1/2}) = -0.5 * x^{-3/2} = -0.5 * y^3
          x->grad[i] += self->grad[i] * (-0.5f * y * y * y);
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// abs(x)
// -----------------------------------------------------------------------------
template <TensorScalar T>
std::shared_ptr<Tensor<T>> abs(const std::shared_ptr<Tensor<T>>& x) {
  bool rg = x->requires_grad;
  auto out = detail::make_result<T>(x->shape, rg, "abs", {x});

  for (std::size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = static_cast<T>(std::abs(static_cast<double>(x->data[i])));
  }

  if (rg) {
    auto x_data = x->data;
    out->backward_fn = [x, x_data]() {
      auto* self = static_cast<Tensor<T>*>(get_current_backward_tensor());
      if (!self) return;
      detail::ensure_grad(*x);

      if (x->requires_grad) {
        for (std::size_t i = 0; i < self->grad.size(); ++i) {
          float xv = static_cast<float>(x_data[i]);
          float sgn = (xv > 0.0f) ? 1.0f : ((xv < 0.0f) ? -1.0f : 0.0f);
          x->grad[i] += self->grad[i] * sgn;
        }
      }
    };
  }

  return out;
}

// -----------------------------------------------------------------------------
// Explicit Template Instantiations
// -----------------------------------------------------------------------------
template std::shared_ptr<Tensor<float>> sin(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> sin(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> sin(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> cos(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> cos(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> cos(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> tan(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> tan(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> tan(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> asin(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> asin(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> asin(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> acos(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> acos(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> acos(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> atan(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> atan(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> atan(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> sinh(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> sinh(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> sinh(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> cosh(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> cosh(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> cosh(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> log2(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> log2(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> log2(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> log10(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> log10(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> log10(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> sqrt(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> sqrt(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> sqrt(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> rsqrt(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> rsqrt(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> rsqrt(const std::shared_ptr<Tensor<int>>&);

template std::shared_ptr<Tensor<float>> abs(const std::shared_ptr<Tensor<float>>&);
template std::shared_ptr<Tensor<double>> abs(const std::shared_ptr<Tensor<double>>&);
template std::shared_ptr<Tensor<int>> abs(const std::shared_ptr<Tensor<int>>&);

}  // namespace scorch
