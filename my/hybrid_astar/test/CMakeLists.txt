cmake_minimum_required(VERSION 3.14)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

function(add_hybrid_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${TEST_NAME}
    PRIVATE
      hybrid_astar
      GTest::GTest
      GTest::Main
      Threads::Threads
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

function(add_hybrid_planner_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${TEST_NAME}
    PRIVATE
      hybrid_astar_planner
      GTest::GTest
      GTest::Main
      Threads::Threads
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Collision checker test
add_hybrid_test(test_collision_checker test_collision_checker.cpp)

# A* test
add_hybrid_test(test_a_star test_a_star.cpp)

# Node tests
add_hybrid_test(test_nodehybrid test_nodehybrid.cpp)

# Smoother test
add_hybrid_test(test_smoother test_smoother.cpp)

# Utils test
add_hybrid_test(test_utils test_utils.cpp)

# Goal manager test
add_hybrid_test(test_goal_manager test_goal_manager.cpp)

# Obstacle heuristic test
add_hybrid_test(test_obstacle_heuristic test_obstacle_heuristic.cpp)

# Distance heuristic test
add_hybrid_test(test_distance_heuristic test_distance_heuristic.cpp)

# ESDF + capsule footprint test
add_hybrid_test(test_esdf_collision test_esdf_collision.cpp)
add_hybrid_planner_test(test_esdf_integration test_esdf_integration.cpp)
