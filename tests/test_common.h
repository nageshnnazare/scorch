#ifndef _AG_TEST_COMMON_H_
#define _AG_TEST_COMMON_H_

/**
 * @file test_common.h
 * @brief Common test registration harness and assertion macros.
 */

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"
#include "tensor_ops.h"

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& get_test_registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline bool register_test(const char* name, std::function<void()> fn) {
  get_test_registry().push_back({name, std::move(fn)});
  return true;
}

#define TEST(name)                                            \
  static void test_##name();                                  \
  static bool reg_##name = register_test(#name, test_##name); \
  static void test_##name()

#define EXPECT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      throw std::runtime_error("assertion failed");                           \
    }                                                                         \
  } while (0)

#define EXPECT_NEAR(a, b, tol) EXPECT(std::fabs((a) - (b)) <= (tol))

#endif  // !_AG_TEST_COMMON_H_
