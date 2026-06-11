cmake_minimum_required(VERSION 3.14)
project(smoother_clothoid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ---- dependencies ----
set(_local_eigen3_compat_dir "${CMAKE_CURRENT_SOURCE_DIR}/cmake/eigen3_compat")
if(
  (NOT DEFINED Eigen3_DIR OR Eigen3_DIR STREQUAL "" OR Eigen3_DIR MATCHES "-NOTFOUND$") AND
  EXISTS "${_local_eigen3_compat_dir}/Eigen3Config.cmake")
  set(Eigen3_DIR "${_local_eigen3_compat_dir}" CACHE PATH "Eigen3 compatibility config" FORCE)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

if(NOT TARGET esdf_core)
  set(_esdf_core_dir "${CMAKE_CURRENT_SOURCE_DIR}/../esdf_core")
  if(EXISTS "${_esdf_core_dir}/CMakeLists.txt")
    set(_esdf_core_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
    add_subdirectory("${_esdf_core_dir}" "${CMAKE_CURRENT_BINARY_DIR}/esdf_core" EXCLUDE_FROM_ALL)
    set(BUILD_TESTING ${_esdf_core_build_testing} CACHE INTERNAL "" FORCE)
  else()
    find_package(esdf_core REQUIRED)
  endif()
endif()

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
  esdf_core
  Ceres::ceres
  Eigen3::Eigen
)

install(DIRECTORY include/ DESTINATION include)

# ---- pybind11 module ----
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(py_smoother_clothoid pybind/py_smoother_clothoid.cpp)
  target_link_libraries(py_smoother_clothoid PRIVATE ${PROJECT_NAME})
  install(TARGETS py_smoother_clothoid DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---- nanobind module ----
option(USE_NANOBIND "Build nanobind Python bindings" OFF)
if(USE_NANOBIND)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(nb_smoother_clothoid nanobind/nb_smoother_clothoid.cpp)
  target_link_libraries(nb_smoother_clothoid PRIVATE ${PROJECT_NAME})
  install(TARGETS nb_smoother_clothoid DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
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
