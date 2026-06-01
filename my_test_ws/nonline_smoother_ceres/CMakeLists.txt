cmake_minimum_required(VERSION 3.14)
project(nonline_smoother_ceres)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/eigen3_compat" CACHE PATH "Eigen3 compatibility config" FORCE)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

add_library(nonline_smoother_ceres ceres_nonlinear_smoother.cpp)
target_include_directories(nonline_smoother_ceres PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nonline_smoother_ceres PUBLIC Ceres::ceres Eigen3::Eigen)

add_executable(nonline_smoother_cusp_demo ceres_cusp_demo.cpp)
target_link_libraries(nonline_smoother_cusp_demo PRIVATE nonline_smoother_ceres)

enable_testing()
add_test(NAME nonline_smoother_cusp_demo COMMAND nonline_smoother_cusp_demo)