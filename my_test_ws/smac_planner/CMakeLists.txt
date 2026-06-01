cmake_minimum_required(VERSION 3.14)
project(smac_planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ompl REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

set(SMAC_PLANNER_SOURCES
  src/a_star.cpp
  src/analytic_expansion.cpp
  src/collision_checker.cpp
  src/costmap_downsampler.cpp
  src/distance_heuristic.cpp
  src/node_2d.cpp
  src/node_basic.cpp
  src/node_hybrid.cpp
  src/node_lattice.cpp
  src/obstacle_heuristic.cpp
  src/smoother.cpp
)

# Common library — core algorithms, no ROS wrappers
add_library(smac_planner_common ${SMAC_PLANNER_SOURCES})
target_include_directories(smac_planner_common
  PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_link_libraries(smac_planner_common
  PUBLIC
    ompl::ompl
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

# Hybrid-A* planner plugin
add_library(smac_planner_hybrid src/smac_planner_hybrid.cpp)
target_include_directories(smac_planner_hybrid PUBLIC include)
target_link_libraries(smac_planner_hybrid PUBLIC smac_planner_common)

# 2D planner plugin
add_library(smac_planner_2d src/smac_planner_2d.cpp)
target_include_directories(smac_planner_2d PUBLIC include)
target_link_libraries(smac_planner_2d PUBLIC smac_planner_common)

# State Lattice planner plugin
add_library(smac_planner_lattice src/smac_planner_lattice.cpp)
target_include_directories(smac_planner_lattice PUBLIC include)
target_link_libraries(smac_planner_lattice PUBLIC smac_planner_common)

# Tests
enable_testing()
add_subdirectory(test)
