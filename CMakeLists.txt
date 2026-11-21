# ==============================================================================
# CMakeLists.txt - Build configuration for scorch
# ==============================================================================

cmake_minimum_required(VERSION 3.20)
project(scorch LANGUAGES CXX)

# Enforce C++20 standard across all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# ------------------------------------------------------------------------------
# 1. Core Library (scorch)
# ------------------------------------------------------------------------------
# Static library containing Tensor, TensorView, and differentiable operations
add_library(scorch STATIC
  src/tensor.cc
  src/ops/arithmetic.cc
  src/ops/matrix.cc
  src/ops/activation.cc
  src/ops/reduction.cc
  src/ops/shape.cc
  src/ops/loss.cc
)

target_include_directories(scorch
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Backwards compatibility alias
add_library(tensor ALIAS scorch)

# ------------------------------------------------------------------------------
# 2. Main Executable (scorch_demo -> binary: scorch)
# ------------------------------------------------------------------------------
# Standalone binary demonstrating basic tensor initialization and printing.
add_executable(scorch_demo
  main.cc
)

target_link_libraries(scorch_demo
  PRIVATE
  scorch
)

set_target_properties(scorch_demo
  PROPERTIES
  OUTPUT_NAME "scorch"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# 3. Unit Testing & CTest Integration
# ------------------------------------------------------------------------------
enable_testing()

add_executable(test_scorch
  tests/test_main.cc
  tests/test_core.cc
  tests/test_ops_arithmetic.cc
  tests/test_ops_matrix.cc
  tests/test_ops_activation.cc
  tests/test_ops_reduction.cc
  tests/test_ops_shape.cc
  tests/test_ops_loss.cc
)

target_link_libraries(test_scorch
  PRIVATE
  scorch
)

set_target_properties(test_scorch
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Individual unit tests registered with CTest so each test case reports its own
# individual status, timing, and pass/fail result in `ninja test` or `ctest`.
set(TENSOR_TESTS
  # Core tensor tests
  zeros_shape
  zeros_values
  ones_values
  full_values
  rand_range
  from_happy
  from_mismatch_throws
  indexing_read
  indexing_write
  requires_grad_flag
  size_and_elements
  backward_leaf
  # Original gradient tests
  grad_add
  grad_mul
  grad_matmul
  grad_relu
  grad_transpose
  grad_reduce
  grad_dag_chain
  grad_diamond_graph
  grad_no_grad_operand
  # Original operator overload tests
  operator_add
  operator_mul
  operator_expression_pipeline
  concept_tensor_scalar
  # NEW: Arithmetic
  grad_sub
  grad_neg
  grad_div
  grad_pow
  grad_exp
  grad_log
  # NEW: Scalar-Tensor
  grad_add_scalar
  grad_mul_scalar
  # NEW: Activations
  grad_sigmoid
  grad_tanh
  grad_leaky_relu
  forward_softmax
  # NEW: Reductions
  grad_mean
  grad_max
  grad_sum_dim
  # NEW: Shape / View
  grad_reshape
  grad_flatten
  grad_squeeze
  grad_unsqueeze
  grad_cat
  # NEW: Loss functions
  grad_mse_loss
  grad_cross_entropy
  # NEW: Operator overloads
  operator_sub
  operator_neg
  operator_div
  operator_add_scalar
  operator_mul_scalar
)

foreach(test_name IN LISTS TENSOR_TESTS)
  add_test(NAME ${test_name} COMMAND test_scorch ${test_name})
endforeach()

# Custom convenience target to run tests with live, verbose console output:
#   ninja check  (or make check)
add_custom_target(check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  DEPENDS test_scorch
  USES_TERMINAL
  COMMENT "Running all unit tests with full output..."
)

# ------------------------------------------------------------------------------
# 4. Code Formatting (clang-format)
# ------------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE NAMES clang-format)

# Collect only project sources (excludes build directory)
file(GLOB_RECURSE ALL_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc
)

if(CLANG_FORMAT_EXE)
  # format: in-place source rewrite
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i --style=file ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format (in-place)..."
    VERBATIM
  )

  # format-check: dry-run, exits non-zero if formatting differs (useful in CI)
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror --style=file ${ALL_SOURCE_FILES}
    COMMENT "Checking clang-format (dry-run)..."
    VERBATIM
  )
else()
  message(WARNING "clang-format not found — 'format' and 'format-check' targets are disabled.")
endif()
