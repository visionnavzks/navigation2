cmake_minimum_required(VERSION 3.14)
project(kinematic_smoother CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)  # significant Ceres optimization speedup

# --- Dependencies ---
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# --- Library ---
add_library(kinematic_smoother_lib INTERFACE)
target_include_directories(kinematic_smoother_lib INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(kinematic_smoother_lib INTERFACE
  Ceres::ceres
  Eigen3::Eigen
)

# --- Standalone executable (JSON I/O via stdin/stdout) ---
add_executable(kinematic_smoother_node src/main.cpp)
target_link_libraries(kinematic_smoother_node PRIVATE kinematic_smoother_lib)

install(TARGETS kinematic_smoother_node DESTINATION bin)
