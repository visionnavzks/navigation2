cmake_minimum_required(VERSION 3.8)
project(costmap_2d)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

set(HEADER_FILES
  include/costmap_2d/point.hpp
  include/costmap_2d/cost_values.hpp
  include/costmap_2d/costmap_math.hpp
  include/costmap_2d/exceptions.hpp
  include/costmap_2d/costmap_2d.hpp
  include/costmap_2d/distance_transform.hpp
  include/costmap_2d/footprint.hpp
  include/costmap_2d/footprint_collision_checker.hpp
  include/costmap_2d/costmap_layer_algorithms.hpp
  include/costmap_2d/inflation_layer_core.hpp
  include/costmap_2d/layer_interface.hpp
  include/costmap_2d/layered_costmap.hpp
  include/costmap_2d/costmap_filters/filter_values.hpp
  include/costmap_2d/denoise/image.hpp
  include/costmap_2d/denoise/image_processing.hpp
)

set(SOURCE_FILES
  src/costmap_2d.cpp
  src/costmap_math.cpp
  src/footprint.cpp
  src/layered_costmap.cpp
)

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
)

# Optional OpenMP support for distance transform
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/costmap_2d
  DESTINATION include
)

# Testing
option(BUILD_TESTING "Build testing" ON)
if(BUILD_TESTING)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    # Try to find GTest manually
    find_path(GTEST_INCLUDE_DIRS gtest/gtest.h)
    find_library(GTEST_LIBRARIES gtest)
    find_library(GTEST_MAIN_LIBRARIES gtest_main)
    if(GTEST_INCLUDE_DIRS AND GTEST_LIBRARIES)
      set(GTest_FOUND TRUE)
      set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
      set(GTEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)
    endif()
  endif()

  if(GTest_FOUND)
    enable_testing()
    include_directories(${GTEST_INCLUDE_DIRS})

    set(TEST_FILES
      test/costmap_test.cpp
      test/costmap_2d_test.cpp
      test/footprint_test.cpp
      test/collision_checker_test.cpp
      test/image_test.cpp
      test/image_processing_test.cpp
      test/distance_transform_test.cpp
      test/layer_algorithms_test.cpp
      test/layered_costmap_test.cpp
    )

    foreach(TEST_FILE ${TEST_FILES})
      get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
      add_executable(${TEST_NAME} ${TEST_FILE})
      target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME} ${GTEST_LIBRARIES})
      target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
      add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
  else()
    message(Warning "GTest not found, tests will not be built")
  endif()
endif()
