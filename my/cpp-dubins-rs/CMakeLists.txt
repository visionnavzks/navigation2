cmake_minimum_required(VERSION 3.16)

project(steering_functions_lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(steering_functions_lite
    src/math_utils.cpp
    src/state_space.cpp
    src/dubins_state_space.cpp
    src/reeds_shepp_state_space.cpp
    src/steering_path.cpp
)

target_include_directories(steering_functions_lite
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(steering_functions_lite PUBLIC cxx_std_17)

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
