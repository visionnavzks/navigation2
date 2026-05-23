set(_EIGEN3_SEARCH_PATHS)

if(DEFINED ENV{EIGEN3_ROOT})
  list(APPEND _EIGEN3_SEARCH_PATHS "$ENV{EIGEN3_ROOT}")
endif()

if(DEFINED ENV{HOMEBREW_PREFIX})
  list(APPEND _EIGEN3_SEARCH_PATHS "$ENV{HOMEBREW_PREFIX}/include")
endif()

list(APPEND _EIGEN3_SEARCH_PATHS /opt/homebrew/include /usr/local/include)

find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core signature_of_eigen3_matrix_library
  PATH_SUFFIXES eigen3 include/eigen3
  PATHS ${_EIGEN3_SEARCH_PATHS}
)

if(NOT EIGEN3_INCLUDE_DIR)
  set(Eigen3_FOUND FALSE)
  set(EIGEN3_FOUND FALSE)
  message(FATAL_ERROR "Failed to locate Eigen headers for the local Eigen3 compatibility config")
endif()

if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()

set(Eigen3_FOUND TRUE)
set(EIGEN3_FOUND TRUE)
set(Eigen3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
set(Eigen3_VERSION "3.4.0")
set(EIGEN3_VERSION_STRING "3.4.0")

unset(_EIGEN3_SEARCH_PATHS)