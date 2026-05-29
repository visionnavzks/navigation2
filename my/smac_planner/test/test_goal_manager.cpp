#include <gtest/gtest.h>
#include "my/smac_planner/goal_manager.hpp"
#include "my/smac_planner/node_2d.hpp"
#include "my/smac_planner/collision_checker.hpp"
#include "my/smac_planner/costmap_2d.hpp"
#include "my/smac_planner/constants.hpp"

using namespace smac_planner;

class GoalManagerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    costmap = std::make_unique<Costmap2D>(10, 10, 1.0, 0.0, 0.0);
    checker = std::make_unique<GridCollisionChecker>(costmap.get(), 1);
    checker->setFootprint(Footprint(), true, 0.0);

    search_info.cost_penalty = 1.0;
    wx_size = 10;
    wy_size = 10;
    wdim3 = 1;
    Node2D::initMotionModel(&ctx, MotionModel::TWOD, wx_size, wy_size, wdim3, search_info);

    for (unsigned int i = 0; i < 5; ++i) {
      nodes.emplace_back(Node2D::getIndex(i, 0, 10), &ctx);
      nodes.back().setPose(Node2D::Coordinates(static_cast<float>(i), 0.0f));
    }

    goal_manager.setContext(&ctx);
  }

  std::unique_ptr<Costmap2D> costmap;
  std::unique_ptr<GridCollisionChecker> checker;
  Node2D::NodeContext ctx;
  SearchInfo search_info;
  unsigned int wx_size{10};
  unsigned int wy_size{10};
  unsigned int wdim3{1};
  std::vector<Node2D> nodes;
  GoalManager<Node2D> goal_manager;
};

TEST_F(GoalManagerTest, AddAndRemoveInvalidGoals) {
  for (auto & node : nodes) {
    Node2D * ptr = &node;
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
    Node2D * ptr = &node;
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
    Node2D * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);

  EXPECT_EQ(goal_manager.getGoalsSet().size(), 4u);
  EXPECT_FALSE(goal_manager.isGoal(&nodes[1]));
  EXPECT_TRUE(goal_manager.isGoal(&nodes[0]));
}

TEST_F(GoalManagerTest, ClearResetsState) {
  for (auto & node : nodes) {
    Node2D * ptr = &node;
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
    Node2D * ptr = &node;
    goal_manager.addGoal(ptr);
  }

  goal_manager.removeInvalidGoals(0.0, checker.get(), true);

  Node2D::NodeVector coarse, fine;
  goal_manager.prepareGoalsForAnalyticExpansion(coarse, fine, 2);

  EXPECT_GT(coarse.size() + fine.size(), 0u);
}
