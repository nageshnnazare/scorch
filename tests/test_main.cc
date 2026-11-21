/**
 * @file test_main.cc
 * @brief Test runner entry point for the test_tensor executable.
 *
 * @details
 * Dispatches test execution based on CLI arguments:
 * - `--list`: Lists all registered test case names to standard output.
 * - `<name>`: Runs a single named test case.
 * - (No args): Executes all registered unit tests sequentially, printing PASS/FAIL
 *   status and returning 0 on success or 1 if any test fails.
 */

#include "test_common.h"

/**
 * @brief Main test runner function.
 * @param argc Command-line argument count.
 * @param argv Command-line arguments.
 * @return 0 on success (all tests pass), non-zero if any test fails or argument is invalid.
 */
int main(int argc, char* argv[]) {
  auto& tests = get_test_registry();

  if (argc > 1) {
    std::string arg = argv[1];

    if (arg == "--list") {
      for (const auto& tc : tests) {
        std::printf("%s\n", tc.name);
      }
      return 0;
    }

    for (const auto& tc : tests) {
      if (arg == tc.name) {
        try {
          tc.fn();
          std::printf("  PASS  %s\n", tc.name);
          return 0;
        } catch (...) {
          std::printf("  FAIL  %s\n", tc.name);
          return 1;
        }
      }
    }

    std::fprintf(stderr, "Unknown test: %s\n", arg.c_str());
    return 1;
  }

  int passed = 0, failed = 0;
  for (auto& tc : tests) {
    try {
      tc.fn();
      std::printf("  PASS  %s\n", tc.name);
      ++passed;
    } catch (...) {
      std::printf("  FAIL  %s\n", tc.name);
      ++failed;
    }
  }
  std::printf("\n%d / %d tests passed\n", passed, passed + failed);
  return failed ? 1 : 0;
}
