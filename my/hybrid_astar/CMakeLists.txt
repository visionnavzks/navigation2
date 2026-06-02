cmake_minimum_required(VERSION 3.14)
project(hybrid_astar)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── steering_functions_lite (Dubins / Reeds-Shepp, replaces OMPL) ────
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp-dubins-rs
                 ${CMAKE_CURRENT_BINARY_DIR}/steering_functions_lite)

# ── esdf_core (shared costmap + ESDF, no Ceres dep) ──────────────────
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

set(HYBRID_ASTAR_SOURCES
  src/a_star.cpp
  src/analytic_expansion.cpp
  src/collision_checker.cpp
  src/costmap_downsampler.cpp
  src/distance_heuristic.cpp
  src/node_basic.cpp
  src/node_hybrid.cpp
  src/obstacle_heuristic.cpp
  src/smoother.cpp
)

add_library(hybrid_astar ${HYBRID_ASTAR_SOURCES})
target_include_directories(hybrid_astar
  PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_link_libraries(hybrid_astar
  PUBLIC
    steering_functions_lite
    esdf_core
)

# Planner plugin
add_library(hybrid_astar_planner src/smac_planner_hybrid.cpp)
target_include_directories(hybrid_astar_planner PUBLIC include)
target_link_libraries(hybrid_astar_planner PUBLIC hybrid_astar)

# ---- pybind11 module ----
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(py_hybrid_astar pybind/py_hybrid_astar.cpp)
  target_link_libraries(py_hybrid_astar PRIVATE hybrid_astar_planner)
  # Hide every symbol contributed by the static dependency libraries (esdf_core,
  # hybrid_astar, hybrid_astar_planner, steering_functions_lite, robin_hood, ...)
  # so they cannot collide with copies of the same symbols pulled in by sibling
  # pybind11 modules (notably py_constrained_smoother) when both are loaded into
  # the same Python interpreter. Without this flag, the second module loaded
  # crashes inside esdf_core because the dynamic linker resolves duplicated weak
  # symbols across modules with mismatched object layouts.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(py_hybrid_astar PRIVATE LINKER:--exclude-libs,ALL)
  endif()
  set_target_properties(py_hybrid_astar PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  install(TARGETS py_hybrid_astar DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
