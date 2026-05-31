#include <gtest/gtest.h>
#include "my/hybrid_astar/goal_manager.hpp"
#include "my/hybrid_astar/node_hybrid.hpp"
#include "my/hybrid_astar/collision_checker.hpp"
#include "my/hybrid_astar/costmap_2d.hpp"
#include "my/hybrid_astar/constants.hpp"

using namespace hybrid_astar;

class GoalManagerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    costmap = std::make_unique<Costmap2D>(10, 10, 1.0, 0.0, 0.0);
    checker = std::make_unique<GridCollisionChecker>(costmap.get(), 72);
    checker->setFootprint(Footprint(), true, 0.0);

    search_info.cost_penalty = 1.0;
    search_info.minimum_turning_radius = 2.0;
    wx_size = 10;
    wy_size = 10;
    wdim3 = 72;
    NodeHybrid::initMotionModel(&ctx, MotionModel::DUBIN, wx_size, wy_size, wdim3, search_info);

    for (unsigned int i = 0; i < 5; ++i) {
      nodes.emplace_back(NodeHybrid::getIndex(i, 0, 0, 10, 72), &ctx);
      nodes.back().setPose(NodeHybrid::Coordinates(static_cast<float>(i), 0.0f, 0.0f));
    }

    goal_manager.setContext(&ctx);
  }

  std::unique_ptr<Costmap2D> costmap;
  std::unique_ptr<GridCollisionChecker> checker;
  NodeHybrid::NodeContext ctx;
  SearchInfo search_info;
  unsigned int wx_size{10};
  unsigned int wy_size{10};
  unsigned int wdim3{72};
  std::vector<NodeHybrid> nodes;
  GoalManager<NodeHybrid> goal_manager;
};

TEST_F(GoalManagerTest, AddAndRemoveInvalidGoals) {
  for (auto & node : nodes) {
    NodeHybrid * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  EXPECT_FALSE(goal_manager.goalsIsEmpty());
  EXPECT_EQ(goal_manager.getGoalsState().size(), 5u);

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);

  EXPECT_EQ(goal_manager.getGoalsSet().size(), 5u);
  EXPECT_EQ(goal_manager.getGoalsCoordinates().size(), 5u);
}

TEST_F(GoalManagerTest, IsGoalAfterRemoveInvalid) {
  for (auto & node : nodes) {
    NodeHybrid * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);

  for (auto & node : nodes) {
    EXPECT_TRUE(goal_manager.isGoal(&node));
  }
}

TEST_F(GoalManagerTest, ObstacleGoalIsRemoved) {
  costmap->setCost(1, 0, OCCUPIED_COST);

  for (auto & node : nodes) {
    NodeHybrid * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);

  EXPECT_EQ(goal_manager.getGoalsSet().size(), 4u);
  EXPECT_FALSE(goal_manager.isGoal(&nodes[1]));
  EXPECT_TRUE(goal_manager.isGoal(&nodes[0]));
}

TEST_F(GoalManagerTest, ClearResetsState) {
  for (auto & node : nodes) {
    NodeHybrid * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);
  EXPECT_EQ(goal_manager.getGoalsSet().size(), 5u);

  goal_manager.clear();
  EXPECT_TRUE(goal_manager.goalsIsEmpty());
  EXPECT_EQ(goal_manager.getGoalsSet().size(), 0u);
  EXPECT_EQ(goal_manager.getGoalsCoordinates().size(), 0u);
}

TEST_F(GoalManagerTest, PrepareGoalsForAnalyticExpansion) {
  for (auto & node : nodes) {
    NodeHybrid * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);

  NodeHybrid::NodeVector coarse, fine;
  goal_manager.prepareGoalsForAnalyticExpansion(coarse, fine, 2);

  EXPECT_GT(coarse.size() + fine.size(), 0u);
}
