cmake_minimum_required(VERSION 3.14)
project(hybrid_astar)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

# ── steering_functions_lite (replaces OMPL Dubins / Reeds-Shepp) ──────
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp-dubins-rs
                 ${CMAKE_CURRENT_BINARY_DIR}/steering_functions_lite)

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
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

# Planner plugin
add_library(hybrid_astar_planner src/smac_planner_hybrid.cpp)
target_include_directories(hybrid_astar_planner PUBLIC include)
target_link_libraries(hybrid_astar_planner PUBLIC hybrid_astar)

# Tests
enable_testing()
add_subdirectory(test)
