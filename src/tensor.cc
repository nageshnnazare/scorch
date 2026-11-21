/**
 * @file tensor.cc
 * @brief Implementation of Tensor member functions, indexing, and autograd traversal.
 */

#include "tensor.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace scorch {

// -----------------------------------------------------------------------------
// Autograd Execution Context
// -----------------------------------------------------------------------------

static thread_local void* g_current_backward_tensor = nullptr;

void* get_current_backward_tensor() { return g_current_backward_tensor; }

// -----------------------------------------------------------------------------
// Shape Utility Functions
// -----------------------------------------------------------------------------

std::size_t numel(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1),
                         std::multiplies<>());
}

std::size_t flatid(const Shape& shape, const std::vector<std::size_t>& coords) {
  assert(coords.size() == shape.size());
  std::size_t id = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    assert(coords[i] < shape[i]);
    id = id * shape[i] + coords[i];
  }
  return id;
}

// -----------------------------------------------------------------------------
// Tensor Subscript Operators
// -----------------------------------------------------------------------------

template <TensorScalar T>
TensorView<T> Tensor<T>::operator[](std::size_t i) {
  std::size_t stride = 1;
  for (std::size_t d = 1; d < shape.size(); ++d) stride *= shape[d];
  return {data.data(), &shape, 1, i * stride};
}

template <TensorScalar T>
const TensorView<const T> Tensor<T>::operator[](std::size_t i) const {
  std::size_t stride = 1;
  for (std::size_t d = 1; d < shape.size(); ++d) stride *= shape[d];
  return {data.data(), &shape, 1, i * stride};
}

// -----------------------------------------------------------------------------
// Tensor Factory Methods
// -----------------------------------------------------------------------------

template <TensorScalar T>
std::shared_ptr<Tensor<T>> Tensor<T>::full(Shape shape, T fill_value, bool requires_grad) {
  auto t = std::make_shared<Tensor<T>>();
  t->shape = std::move(shape);
  t->data.assign(numel(t->shape), fill_value);
  t->requires_grad = requires_grad;
  if (requires_grad) {
    t->grad.assign(t->data.size(), 0.f);
  }
  return t;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> Tensor<T>::zeros(Shape shape, bool requires_grad) {
  return full(std::move(shape), T(0), requires_grad);
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> Tensor<T>::ones(Shape shape, bool requires_grad) {
  return full(std::move(shape), T(1), requires_grad);
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> Tensor<T>::rand(Shape shape, T min, T max, bool requires_grad) {
  auto t = std::make_shared<Tensor<T>>();
  t->shape = std::move(shape);
  t->data.resize(numel(t->shape));
  t->requires_grad = requires_grad;
  if (requires_grad) {
    t->grad.assign(t->data.size(), 0.f);
  }

  // Thread-local pseudo-random number generator for uniform distribution in [min, max)
  thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<double> dist(static_cast<double>(min), static_cast<double>(max));

  for (auto& v : t->data) v = static_cast<T>(dist(rng));

  return t;
}

template <TensorScalar T>
std::shared_ptr<Tensor<T>> Tensor<T>::from(Shape shape, std::vector<T> data, bool requires_grad) {
  const std::size_t n = numel(shape);
  if (data.size() != n) {
    throw std::invalid_argument("scorch::Tensor::from — data size (" + std::to_string(data.size()) +
                                ") does not match shape numel (" + std::to_string(n) + ")");
  }

  auto t = std::make_shared<Tensor<T>>();
  t->shape = std::move(shape);
  t->data = std::move(data);
  t->requires_grad = requires_grad;
  if (requires_grad) {
    t->grad.assign(t->data.size(), 0.f);
  }
  return t;
}

// -----------------------------------------------------------------------------
// Autograd & Backward Graph Traversal
// -----------------------------------------------------------------------------

template <TensorScalar T>
void Tensor<T>::zero_grad() {
  if (!grad.empty()) {
    std::fill(grad.begin(), grad.end(), 0.f);
  }
}

template <TensorScalar T>
void Tensor<T>::backward() {
  if (grad.size() != data.size()) {
    grad.assign(data.size(), 0.f);
  }
  if (shape.empty()) {
    grad[0] = 1.f;
  } else {
    // Seed gradient: dL/dL = 1
    std::fill(grad.begin(), grad.end(), 1.f);
  }

  // Collect topological ordering of DAG nodes
  std::vector<Tensor*> order;
  std::unordered_set<Tensor*> seen;

  std::function<void(Tensor*)> topo = [&](Tensor* t) {
    if (!seen.insert(t).second) {
      return;
    }
    for (auto& p : t->parents) {
      topo(p.get());
    }
    order.push_back(t);
  };

  topo(this);

  // Traverse in reverse topological order, calling each backward function
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    Tensor* t = *it;
    if (t->backward_fn) {
      g_current_backward_tensor = t;
      t->backward_fn();
      g_current_backward_tensor = nullptr;
    }
  }
}

// -----------------------------------------------------------------------------
// Formatting & Pretty Printing
// -----------------------------------------------------------------------------

template <TensorScalar T>
void Tensor<T>::print(const char* name) const {
  std::size_t indent = std::strlen(name) + 1;
  std::string pad(indent, ' ');

  std::printf("%s shape=[", name);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) std::printf(",");
    std::printf("%zu", shape[i]);
  }
  std::printf("]\n");
  std::printf("%sdata=", pad.c_str());
  print_dim(0, 0, pad.c_str());
  std::printf("\n");
}

template <TensorScalar T>
void Tensor<T>::print_dim(int dim, std::size_t offset, const char* base_indent) const {
  if (dim == static_cast<int>(shape.size())) {
    std::printf("%.6g", static_cast<double>(data[offset]));
    return;
  }

  std::printf("[");

  std::size_t stride = 1;
  for (int d = dim + 1; d < static_cast<int>(shape.size()); ++d) stride *= shape[d];

  for (std::size_t i = 0; i < shape[dim]; ++i) {
    if (i) std::printf(",");
    if (dim + 1 < static_cast<int>(shape.size())) {
      std::printf("\n");
      std::printf("%s", base_indent);
      for (int d = 0; d <= dim; ++d) std::printf("  ");
    }
    print_dim(dim + 1, offset + i * stride, base_indent);
  }

  if (dim + 1 < static_cast<int>(shape.size()) && shape[dim] > 0) {
    std::printf("\n");
    std::printf("%s", base_indent);
    for (int d = 0; d < dim; ++d) std::printf("  ");
  }

  std::printf("]");
}

// -----------------------------------------------------------------------------
// Explicit Template Instantiations
// -----------------------------------------------------------------------------

template struct Tensor<float>;
template struct Tensor<double>;
template struct Tensor<int>;

}  // namespace scorch
