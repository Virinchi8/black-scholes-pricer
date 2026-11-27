cmake_minimum_required(VERSION 3.16)
project(bsm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BSM_BUILD_TESTS "Build the test suite" ON)
option(BSM_BUILD_BENCH "Build the throughput benchmark" ON)

add_library(bsm STATIC src/black_scholes.cpp)
target_include_directories(bsm PUBLIC include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(bsm PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(bsm_cli src/main.cpp)
target_link_libraries(bsm_cli PRIVATE bsm)

if(BSM_BUILD_BENCH)
  add_executable(bsm_bench bench/bench.cpp)
  target_link_libraries(bsm_bench PRIVATE bsm)
endif()

if(BSM_BUILD_TESTS)
  enable_testing()
  add_executable(bsm_tests tests/test_black_scholes.cpp)
  target_link_libraries(bsm_tests PRIVATE bsm)
  # 50-digit reference values shared with the Python tests.
  target_compile_definitions(bsm_tests PRIVATE
    BSM_REFERENCE_CSV="${CMAKE_CURRENT_SOURCE_DIR}/../data/reference.csv")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bsm_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME bsm_tests COMMAND bsm_tests)
endif()
