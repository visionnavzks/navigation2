cmake_minimum_required(VERSION 3.14)
project(ceres_smoother_2d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aggressive optimization for real-time path planning
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Dependencies
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

# stb_image (header-only, bundled)
set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb")

# stb_image implementation (separate TU, needs fPIC for shared libs)
add_library(stb_image_impl STATIC src/stb_image_impl.cpp)
target_include_directories(stb_image_impl PRIVATE ${STB_INCLUDE_DIR})
set_target_properties(stb_image_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Main library (header-only)
add_library(ceres_smoother_2d_headers INTERFACE)
target_include_directories(ceres_smoother_2d_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STB_INCLUDE_DIR}
)
target_link_libraries(ceres_smoother_2d_headers INTERFACE
  Ceres::ceres
  Eigen3::Eigen
)

# Demo executable
add_executable(ceres_smoother_2d_demo src/main.cpp)
target_link_libraries(ceres_smoother_2d_demo PRIVATE ceres_smoother_2d_headers stb_image_impl)

# nanobind Python module
set(Python_EXECUTABLE "/home/zks/.venv/bin/python3")
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind from the venv installation
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__))"
  OUTPUT_VARIABLE NANOBIND_DIR_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(nanobind_DIR "${NANOBIND_DIR_RAW}/cmake")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(ceres_smoother_2d src/nanobind_module.cpp)
target_link_libraries(ceres_smoother_2d PRIVATE ceres_smoother_2d_headers stb_image_impl)
target_include_directories(ceres_smoother_2d PRIVATE ${STB_INCLUDE_DIR})

# Install
install(TARGETS ceres_smoother_2d_demo DESTINATION bin)
install(TARGETS ceres_smoother_2d LIBRARY DESTINATION .)

# ========================================================================
# Tests
# ========================================================================
enable_testing()

# C++ unit tests (header-only library, so just compile + link)
add_executable(ceres_smoother_2d_tests tests/test_cpp.cpp)
target_link_libraries(ceres_smoother_2d_tests PRIVATE ceres_smoother_2d_headers stb_image_impl)
target_include_directories(ceres_smoother_2d_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STB_INCLUDE_DIR}
)
add_test(NAME ceres_smoother_2d_cpp_tests
         COMMAND ceres_smoother_2d_tests)
