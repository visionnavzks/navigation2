cmake_minimum_required(VERSION 3.14)
project(kinematic_path_smoother)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

if(NOT TARGET esdf_core)
  set(_esdf_core_dir "${CMAKE_CURRENT_SOURCE_DIR}/../esdf_core")
  if(EXISTS "${_esdf_core_dir}/CMakeLists.txt")
    set(_saved_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
    add_subdirectory("${_esdf_core_dir}" "${CMAKE_CURRENT_BINARY_DIR}/esdf_core" EXCLUDE_FROM_ALL)
    set(BUILD_TESTING ${_saved_build_testing} CACHE INTERNAL "" FORCE)
  else()
    find_package(esdf_core REQUIRED)
  endif()
endif()

if(${CERES_VERSION} VERSION_LESS_EQUAL 2.0.0)
  add_compile_definitions(USE_OLD_CERES_API)
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME}
  INTERFACE
    Ceres::ceres
    Eigen3::Eigen
    esdf_core)

install(DIRECTORY include/ DESTINATION include)

option(BUILD_PYTHON "Build nanobind Python bindings for the web demo" OFF)
if(BUILD_PYTHON)
  if(NOT DEFINED Python_EXECUTABLE AND EXISTS "/home/zks/.venv/bin/python3")
    set(Python_EXECUTABLE "/home/zks/.venv/bin/python3" CACHE FILEPATH "Python executable for nanobind" FORCE)
  endif()
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__))"
    OUTPUT_VARIABLE NANOBIND_DIR_RAW
    RESULT_VARIABLE NANOBIND_IMPORT_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT NANOBIND_IMPORT_RESULT EQUAL 0)
    message(FATAL_ERROR "nanobind is required for BUILD_PYTHON. Install it for ${Python_EXECUTABLE}.")
  endif()
  set(nanobind_DIR "${NANOBIND_DIR_RAW}/cmake")
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(py_kinematic_path_smoother nanobind/nb_kinematic_path_smoother.cpp)
  target_link_libraries(py_kinematic_path_smoother PRIVATE ${PROJECT_NAME})
  install(TARGETS py_kinematic_path_smoother DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

option(BUILD_TESTS "Build kinematic_path_smoother tests" ON)
if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)

  add_executable(test_kinematic_path_smoother test/test_kinematic_path_smoother.cpp)
  target_link_libraries(test_kinematic_path_smoother
    PRIVATE
      ${PROJECT_NAME}
      GTest::gtest
      GTest::gtest_main)
  add_test(NAME test_kinematic_path_smoother COMMAND test_kinematic_path_smoother)
endif()
