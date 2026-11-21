#ifndef _SCORCH_TEST_COMMON_H_
#define _SCORCH_TEST_COMMON_H_

/**
 * @file test_common.h
 * @brief Common test registration harness and assertion macros for scorch unit tests.
 *
 * @details
 * Provides a lightweight macro-based test registration and execution harness
 * inspired by GoogleTest and Catch2 without external library dependencies:
 * - `TEST(name)`: Declares a self-registering test function.
 * - `EXPECT(cond)`: Verifies boolean condition; prints file/line and throws runtime_error on failure.
 * - `EXPECT_NEAR(a, b, tol)`: Verifies floating-point closeness within tolerance.
 *
 * @ingroup tests
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

/**
 * @brief Descriptor representing a registered unit test case.
 */
struct TestCase {
  const char* name;          ///< Human-readable test identifier.
  std::function<void()> fn;  ///< Invocation callback executing the test logic.
};

/**
 * @brief Global registry of all registered unit tests.
 * @return Reference to static vector of TestCase objects.
 */
inline std::vector<TestCase>& get_test_registry() {
  static std::vector<TestCase> tests;
  return tests;
}

/**
 * @brief Registers a test case with the global test registry.
 * @param name Test name identifier string.
 * @param fn Callable function executing the test assertions.
 * @return Always returns true to allow static boolean initialization.
 */
inline bool register_test(const char* name, std::function<void()> fn) {
  get_test_registry().push_back({name, std::move(fn)});
  return true;
}

/**
 * @def TEST(name)
 * @brief Defines a named test function that registers itself automatically at program startup.
 * @param name Unique identifier for the test case.
 */
#define TEST(name)                                            \
  static void test_##name();                                  \
  static bool reg_##name = register_test(#name, test_##name); \
  static void test_##name()

/**
 * @def EXPECT(cond)
 * @brief Asserts that condition `cond` evaluates to true; throws std::runtime_error on failure.
 * @param cond Boolean expression.
 */
#define EXPECT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      throw std::runtime_error("assertion failed");                           \
    }                                                                         \
  } while (0)

/**
 * @def EXPECT_NEAR(a, b, tol)
 * @brief Asserts that floating point value `a` is within `tol` of `b`.
 * @param a First value.
 * @param b Second value.
 * @param tol Maximum allowed absolute error.
 */
#define EXPECT_NEAR(a, b, tol) EXPECT(std::fabs((a) - (b)) <= (tol))

#endif  // !_SCORCH_TEST_COMMON_H_
