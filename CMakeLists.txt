# ==============================================================================
# CMakeLists.txt - Build configuration for autograd_cpp
# ==============================================================================

cmake_minimum_required(VERSION 3.20)
project(autograd_cpp LANGUAGES CXX)

# Enforce C++20 standard across all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# ------------------------------------------------------------------------------
# 1. Core Library (tensor)
# ------------------------------------------------------------------------------
# Static library containing Tensor, TensorView, and differentiable operations
add_library(tensor STATIC
  src/tensor.cc
  src/tensor_ops.cc
)

target_include_directories(tensor
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------------------------
# 2. Main Executable (autograd)
# ------------------------------------------------------------------------------
# Standalone binary demonstrating basic tensor initialization and printing.
add_executable(autograd
  main.cc
)

target_link_libraries(autograd
  PRIVATE
  tensor
)

set_target_properties(autograd
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# 3. Unit Testing & CTest Integration
# ------------------------------------------------------------------------------
enable_testing()

add_executable(test_tensor
  tests/test_tensor.cc
)

target_link_libraries(test_tensor
  PRIVATE
  tensor
)

set_target_properties(test_tensor
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Individual unit tests registered with CTest so each test case reports its own
# individual status, timing, and pass/fail result in `ninja test` or `ctest`.
set(TENSOR_TESTS
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
  grad_add
  grad_mul
  grad_matmul
  grad_relu
  grad_transpose
  grad_reduce
  grad_dag_chain
  grad_diamond_graph
  grad_no_grad_operand
  operator_add
  operator_mul
  operator_expression_pipeline
  concept_tensor_scalar
)

foreach(test_name IN LISTS TENSOR_TESTS)
  add_test(NAME ${test_name} COMMAND test_tensor ${test_name})
endforeach()

# Custom convenience target to run tests with live, verbose console output:
#   ninja check  (or make check)
add_custom_target(check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  DEPENDS test_tensor
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
