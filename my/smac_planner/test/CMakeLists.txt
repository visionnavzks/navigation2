cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Helper to add a test
function(add_smac_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${TEST_NAME}
    PRIVATE
      smac_planner_common
      GTest::GTest
      GTest::Main
      Threads::Threads
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Collision checker test
add_smac_test(test_collision_checker test_collision_checker.cpp)

# A* test
add_smac_test(test_a_star test_a_star.cpp)

# Node tests
add_smac_test(test_node2d test_node2d.cpp)
add_smac_test(test_nodehybrid test_nodehybrid.cpp)
add_smac_test(test_nodelattice test_nodelattice.cpp)

# Smoother test
add_smac_test(test_smoother test_smoother.cpp)

# Utils test
add_smac_test(test_utils test_utils.cpp)

# Goal manager test
add_smac_test(test_goal_manager test_goal_manager.cpp)
