#include <gtest/gtest.h>
#include "my/hybrid_astar/smoother.hpp"
#include "my/hybrid_astar/costmap_2d.hpp"

using namespace hybrid_astar;

TEST(SmootherTest, SmoothStraightLine) {
  SmootherParams params;
  params.get(1e-3, 200, 0.3, 0.3, false, 3);

  Smoother smoother(params);
  smoother.initialize(2.0);

  Costmap2D costmap(100, 100, 0.1, -5.0, -5.0);

  Path path;
  const int N = 20;
  for (int i = 0; i < N; i++) {
    double t = static_cast<double>(i) / (N - 1);
    path.push_back({t * 5.0, t * 5.0, 0.0});
  }

  bool success = smoother.smooth(path, &costmap, 2.0);
  EXPECT_TRUE(success);
  EXPECT_EQ(path.size(), static_cast<size_t>(N));

  EXPECT_FLOAT_EQ(path.front().x, 0.0);
  EXPECT_FLOAT_EQ(path.front().y, 0.0);
  EXPECT_FLOAT_EQ(path.back().x, 5.0);
  EXPECT_FLOAT_EQ(path.back().y, 5.0);
}

TEST(SmootherTest, SmoothEmptyPathTriviallySucceeds) {
  SmootherParams params;
  params.get(1e-3, 200, 0.3, 0.3, false, 3);

  Smoother smoother(params);
  smoother.initialize(2.0);

  Costmap2D costmap(100, 100, 0.1, -5.0, -5.0);
  Path path;

  bool success = smoother.smooth(path, &costmap, 2.0);
  EXPECT_TRUE(success);
  EXPECT_EQ(path.size(), 0u);
}

TEST(SmootherTest, SmoothShortPathStillWorks) {
  SmootherParams params;
  params.get(1e-3, 200, 0.3, 0.3, false, 3);

  Smoother smoother(params);
  smoother.initialize(2.0);

  Costmap2D costmap(100, 100, 0.1, -5.0, -5.0);

  Path path;
  path.push_back({0.0, 0.0, 0.0});
  path.push_back({1.0, 0.0, 0.0});

  bool success = smoother.smooth(path, &costmap, 2.0);
  EXPECT_TRUE(success);
  EXPECT_EQ(path.size(), 2u);
}
