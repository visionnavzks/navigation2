#include <gtest/gtest.h>
#include "my/smac_planner/node_2d.hpp"
#include "my/smac_planner/constants.hpp"

using namespace smac_planner;

class Node2DTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    search_info.cost_penalty = 1.0;
    wx_size = 10;
    wy_size = 10;
    wdim3 = 1;
    Node2D::initMotionModel(&ctx, MotionModel::TWOD, wx_size, wy_size, wdim3, search_info);
  }

  Node2D::NodeContext ctx;
  SearchInfo search_info;
  unsigned int wx_size{10};
  unsigned int wy_size{10};
  unsigned int wdim3{1};
};

TEST_F(Node2DTest, InitAndIndex) {
  const unsigned int WIDTH = 10;
  uint64_t node_idx = Node2D::getIndex(3, 4, WIDTH);
  EXPECT_EQ(node_idx, static_cast<uint64_t>(3 + 4 * WIDTH));

  Node2D node(node_idx, &ctx);
  EXPECT_EQ(node.getIndex(), node_idx);

  Node2D::Coordinates coords = Node2D::getCoords(node_idx, WIDTH, 1);
  EXPECT_FLOAT_EQ(coords.x, 3.0f);
  EXPECT_FLOAT_EQ(coords.y, 4.0f);
}

TEST_F(Node2DTest, CostAccumulatedCost) {
  Node2D node(0, &ctx);

  EXPECT_TRUE(std::isnan(node.getCost()));
  EXPECT_FLOAT_EQ(node.getAccumulatedCost(), std::numeric_limits<float>::max());

  node.setCost(100.0f);
  EXPECT_FLOAT_EQ(node.getCost(), 100.0f);

  node.setAccumulatedCost(50.0f);
  EXPECT_FLOAT_EQ(node.getAccumulatedCost(), 50.0f);
}

TEST_F(Node2DTest, VisitedAndQueued) {
  Node2D node(0, &ctx);

  EXPECT_FALSE(node.wasVisited());
  EXPECT_FALSE(node.isQueued());

  node.queued();
  EXPECT_TRUE(node.isQueued());

  node.visited();
  EXPECT_TRUE(node.wasVisited());
  EXPECT_FALSE(node.isQueued());
}

TEST_F(Node2DTest, GetCoordsInstance) {
  uint64_t idx = Node2D::getIndex(7, 2, 10);
  Node2D node(idx, &ctx);
  auto coords = node.getCoords(idx);
  EXPECT_FLOAT_EQ(coords.x, 7.0f);
  EXPECT_FLOAT_EQ(coords.y, 2.0f);
}

TEST_F(Node2DTest, TraversalCost) {
  Node2D parent(Node2D::getIndex(5, 5, 10), &ctx);
  Node2D child_cardinal(Node2D::getIndex(5, 6, 10), &ctx);
  child_cardinal.setCost(FREE_COST);
  float cost = parent.getTraversalCost(&child_cardinal);
  EXPECT_FLOAT_EQ(cost, 1.0f);

  Node2D child_diag(Node2D::getIndex(6, 6, 10), &ctx);
  child_diag.setCost(FREE_COST);
  float diag_cost = parent.getTraversalCost(&child_diag);
  EXPECT_FLOAT_EQ(diag_cost, static_cast<float>(M_SQRT2));
}

TEST_F(Node2DTest, GetIndexStatic) {
  uint64_t idx = Node2D::getIndex(0, 0, 32);
  EXPECT_EQ(idx, 0u);

  idx = Node2D::getIndex(31, 0, 32);
  EXPECT_EQ(idx, 31u);

  idx = Node2D::getIndex(0, 31, 32);
  EXPECT_EQ(idx, 31u * 32u);

  idx = Node2D::getIndex(31, 31, 32);
  EXPECT_EQ(idx, 31u * 32u + 31u);
}

TEST_F(Node2DTest, ResetRestoresState) {
  Node2D node(0, &ctx);
  node.setCost(50.0f);
  node.setAccumulatedCost(25.0f);
  node.visited();

  node.reset();

  EXPECT_TRUE(std::isnan(node.getCost()));
  EXPECT_FLOAT_EQ(node.getAccumulatedCost(), std::numeric_limits<float>::max());
  EXPECT_FALSE(node.wasVisited());
  EXPECT_FALSE(node.isQueued());
}

TEST_F(Node2DTest, GetCoordsStaticThrowsOnNonOneAngles) {
  EXPECT_THROW(Node2D::getCoords(0, 10, 2), std::runtime_error);
}
