cmake_minimum_required(VERSION 3.14)
project(constrained_smoother)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)  # significant Ceres optimization speedup

# ---- dependencies ----
set(_local_eigen3_compat_dir "${CMAKE_CURRENT_SOURCE_DIR}/cmake/eigen3_compat")
if(
  (NOT DEFINED Eigen3_DIR OR Eigen3_DIR STREQUAL "" OR Eigen3_DIR MATCHES "-NOTFOUND$") AND
  EXISTS "${_local_eigen3_compat_dir}/Eigen3Config.cmake")
  set(Eigen3_DIR "${_local_eigen3_compat_dir}" CACHE PATH "Eigen3 compatibility config" FORCE)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

# esdf_core is the shared costmap + ESDF library (no Ceres dep).
# We add it as a subdirectory when it's sitting next to us in the my/ tree.
# When the user has already installed it system-wide, find_package will
# succeed and we fall through to the IMPORTED target.
if(NOT TARGET esdf_core)
  set(_esdf_core_dir "${CMAKE_CURRENT_SOURCE_DIR}/../esdf_core")
  if(EXISTS "${_esdf_core_dir}/CMakeLists.txt")
    # Suppress esdf_core's own tests when embedded; we only need the library
    # interface. (EXCLUDE_FROM_ALL alone still leaves the test registered
    # in ctest, which then reports "Not Run" because the binary is never
    # built by the parent `all` target.)
    set(_esdf_core_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
    add_subdirectory("${_esdf_core_dir}" "${CMAKE_CURRENT_BINARY_DIR}/esdf_core" EXCLUDE_FROM_ALL)
    set(BUILD_TESTING ${_esdf_core_build_testing} CACHE INTERNAL "" FORCE)
  else()
    find_package(esdf_core REQUIRED)
  endif()
endif()

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
  esdf_core
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
