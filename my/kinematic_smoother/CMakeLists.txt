cmake_minimum_required(VERSION 3.14)
project(kinematic_smoother)

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

include_directories(./distance_transform/include)
include_directories(./distance_transform/extern/DopeVector/include)

if(${CERES_VERSION} VERSION_LESS_EQUAL 2.0.0)
  add_compile_definitions(USE_OLD_CERES_API)
endif()

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

install(DIRECTORY include/ DESTINATION include)

option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(py_kinematic_smoother pybind/py_kinematic_smoother.cpp)
  target_link_libraries(py_kinematic_smoother PRIVATE ${PROJECT_NAME})
  install(TARGETS py_kinematic_smoother DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()
