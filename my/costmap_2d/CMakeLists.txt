cmake_minimum_required(VERSION 3.14)
project(costmap_2d_standalone)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()

  add_executable(test_costmap_2d test/test_costmap_2d.cpp)
  target_link_libraries(test_costmap_2d PRIVATE ${PROJECT_NAME})
  add_test(NAME test_costmap_2d COMMAND test_costmap_2d)
endif()

install(DIRECTORY include/ DESTINATION include)