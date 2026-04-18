cmake_minimum_required(VERSION 3.14)
project(constrained_smoother)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)  # significant Ceres optimization speedup

# ---- dependencies ----
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

# Check Ceres version for API compatibility
if(${CERES_VERSION} VERSION_LESS_EQUAL 2.0.0)
  add_compile_definitions(USE_OLD_CERES_API)
endif()

# ---- header-only library target ----
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(${PROJECT_NAME} INTERFACE
  Ceres::ceres
  Eigen3::Eigen
)

# ---- install ----
install(DIRECTORY include/ DESTINATION include)

# ---- pybind11 module ----
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(py_constrained_smoother pybind/py_constrained_smoother.cpp)
  target_link_libraries(py_constrained_smoother PRIVATE ${PROJECT_NAME})
  # Install to the project root so the web app can find it
  install(TARGETS py_constrained_smoother DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---- tests ----
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)

  add_executable(test_smoother test/test_smoother.cpp)
  target_link_libraries(test_smoother PRIVATE
    ${PROJECT_NAME}
    GTest::gtest
    GTest::gtest_main
  )
  add_test(NAME test_smoother COMMAND test_smoother)
endif()
