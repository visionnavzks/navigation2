cmake_minimum_required(VERSION 3.14)
project(esdf_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- header-only library target ----
add_library(${PROJECT_NAME} INTERFACE)

# Expose the public headers and the bundled distance_transform + DopeVector
# include paths so consumers don't need to add them manually.
target_include_directories(${PROJECT_NAME}
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/extern/DopeVector/include>"
)

# Threads for distance_transform's parallel sweep.
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} INTERFACE
  Threads::Threads
)

# ---- install ----
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY thirdparty/distance_transform/include/ DESTINATION include/distance_transform)
install(DIRECTORY thirdparty/distance_transform/extern/DopeVector/include/ DESTINATION include)

# ---- tests ----
option(BUILD_TESTING "Build esdf_core tests" OFF)
if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(test)
endif()
