/**
 * @file test_core.cc
 * @brief Unit tests for core Tensor construction, indexing, and DAG traversal.
 */

#include "test_common.h"

TEST(zeros_shape) {
  auto t = AG::Tensor<float>::zeros({3, 5});
  EXPECT(t->shape.size() == 2);
  EXPECT(t->shape[0] == 3);
  EXPECT(t->shape[1] == 5);
  EXPECT(t->data.size() == 15);
}

TEST(zeros_values) {
  auto t = AG::Tensor<float>::zeros({2, 4});
  for (auto v : t->data) EXPECT_NEAR(v, 0.f, 1e-9f);
}

TEST(ones_values) {
  auto t = AG::Tensor<float>::ones({2, 3});
  for (auto v : t->data) EXPECT_NEAR(v, 1.f, 1e-9f);
}

TEST(full_values) {
  auto t = AG::Tensor<float>::full({3, 3}, 7.f);
  for (auto v : t->data) EXPECT_NEAR(v, 7.f, 1e-9f);
}

TEST(rand_range) {
  auto t = AG::Tensor<float>::rand({10, 10});
  for (auto v : t->data) {
    EXPECT(v >= 0.f);
    EXPECT(v < 1.f);
  }
}

TEST(from_happy) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT(t->shape[0] == 2 && t->shape[1] == 3);
  EXPECT_NEAR(t->data[0], 1.f, 1e-9f);
  EXPECT_NEAR(t->data[5], 6.f, 1e-9f);
}

TEST(from_mismatch_throws) {
  bool threw = false;
  try {
    AG::Tensor<float>::from({2, 3}, {1, 2, 3});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

TEST(indexing_read) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_NEAR(static_cast<float>((*t)[0][0]), 1.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[0][2]), 3.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[1][0]), 4.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[1][2]), 6.f, 1e-9f);
}

TEST(indexing_write) {
  auto t = AG::Tensor<float>::zeros({2, 3});
  (*t)[1][2] = 42.f;
  EXPECT_NEAR(static_cast<float>((*t)[1][2]), 42.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>((*t)[1][1]), 0.f, 1e-9f);
}

TEST(requires_grad_flag) {
  auto a = AG::Tensor<float>::zeros({2}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::zeros({2}, /*requires_grad=*/false);
  EXPECT(a->requires_grad == true);
  EXPECT(b->requires_grad == false);
}

TEST(size_and_elements) {
  AG::Shape shape{2, 3};
  auto t = AG::Tensor<float>::from(shape, {1, 2, 3, 4, 5, 6});
  EXPECT(AG::numel(shape) == 6);
  EXPECT(AG::flatid(shape, {1, 1}) == 4);
  EXPECT(t->data[AG::flatid(shape, {1, 2})] == 6);
}

TEST(backward_leaf) {
  auto t = AG::Tensor<float>::zeros({2, 3}, /*requires_grad=*/true);
  t->backward();
  EXPECT(t->grad.size() == 6);
  for (auto g : t->grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

TEST(grad_dag_chain) {
  auto a = AG::Tensor<float>::full({1}, 2.f, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto c = AG::add(a, b);
  auto d = AG::mul(c, a);
  EXPECT_NEAR(d->data[0], 10.f, 1e-9f);

  d->backward();
  EXPECT_NEAR(a->grad[0], 7.f, 1e-9f);
  EXPECT_NEAR(b->grad[0], 2.f, 1e-9f);
}

TEST(grad_diamond_graph) {
  auto x = AG::Tensor<float>::full({1}, 3.f, /*requires_grad=*/true);
  auto y = AG::mul(x, x);
  EXPECT_NEAR(y->data[0], 9.f, 1e-9f);

  y->backward();
  EXPECT_NEAR(x->grad[0], 6.f, 1e-9f);
}

TEST(grad_no_grad_operand) {
  auto a = AG::Tensor<float>::from({2}, {2.f, 3.f}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::from({2}, {4.f, 5.f}, /*requires_grad=*/false);
  auto c = AG::mul(a, b);

  c->backward();
  EXPECT_NEAR(a->grad[0], 4.f, 1e-9f);
  EXPECT_NEAR(a->grad[1], 5.f, 1e-9f);
  EXPECT(b->grad.empty());
}

TEST(concept_tensor_scalar) {
  static_assert(AG::TensorScalar<float>);
  static_assert(AG::TensorScalar<const float>);
  static_assert(AG::TensorScalar<double>);
  static_assert(AG::TensorScalar<const double>);
  static_assert(AG::TensorScalar<int>);
  static_assert(AG::TensorScalar<const int>);

  static_assert(!AG::TensorScalar<char>);
  static_assert(!AG::TensorScalar<bool>);
  static_assert(!AG::TensorScalar<std::string>);
  EXPECT(true);
}
