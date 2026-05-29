// LayeredCostmap and LayerInterface Tests

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "my_costmap_2d/layered_costmap.hpp"
#include "my_costmap_2d/layer_interface.hpp"
#include "my_costmap_2d/cost_values.hpp"

using namespace my_costmap_2d;

// ===================== Mock Layer =====================

class MockLayer : public LayerInterface
{
public:
  MockLayer()
  : update_bounds_called_(false), update_costs_called_(false),
    min_x_(0), min_y_(0), max_x_(0), max_y_(0) {}

  void reset() override {}

  bool isClearable() override {return false;}

  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y,
    double * max_x, double * max_y) override
  {
    update_bounds_called_ = true;
    // Update bounds to include the robot position (standard behavior)
    *min_x = std::min(robot_x - 1.0, *min_x);
    *min_y = std::min(robot_y - 1.0, *min_y);
    *max_x = std::max(robot_x + 1.0, *max_x);
    *max_y = std::max(robot_y + 1.0, *max_y);
  }

  void updateCosts(
    Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override
  {
    update_costs_called_ = true;
    // Mark some cells
    for (int j = min_j; j < max_j; ++j) {
      for (int i = min_i; i < max_i; ++i) {
        if (i == 5 && j == 5) {
          master_grid.setCost(i, j, LETHAL_OBSTACLE);
        }
      }
    }
  }

  bool update_bounds_called_;
  bool update_costs_called_;
  double min_x_, min_y_, max_x_, max_y_;
};

// ===================== LayeredCostmap Tests =====================

TEST(LayeredCostmap, testConstructor)
{
  LayeredCostmap costmap("map", false, false);
  EXPECT_EQ(costmap.getGlobalFrameID(), "map");
  EXPECT_FALSE(costmap.isRolling());
  EXPECT_FALSE(costmap.isTrackingUnknown());
}

TEST(LayeredCostmap, testTrackUnknown)
{
  LayeredCostmap costmap("map", false, true);
  EXPECT_TRUE(costmap.isTrackingUnknown());
}

TEST(LayeredCostmap, testResizeMap)
{
  LayeredCostmap costmap("map", false, false);
  costmap.resizeMap(20, 20, 0.05, 0.0, 0.0);
  EXPECT_EQ(costmap.getCostmap()->getSizeInCellsX(), 20u);
  EXPECT_EQ(costmap.getCostmap()->getSizeInCellsY(), 20u);
}

TEST(LayeredCostmap, testAddPlugin)
{
  LayeredCostmap costmap("map", false, false);
  auto layer = std::make_shared<MockLayer>();
  costmap.addPlugin(layer);
  EXPECT_EQ(costmap.getPlugins()->size(), 1u);
}

TEST(LayeredCostmap, testAddFilter)
{
  LayeredCostmap costmap("map", false, false);
  auto filter = std::make_shared<MockLayer>();
  costmap.addFilter(filter);
  EXPECT_EQ(costmap.getFilters()->size(), 1u);
}

TEST(LayeredCostmap, testSetFootprint)
{
  LayeredCostmap costmap("map", false, false);
  std::vector<Point> footprint;
  footprint.push_back({-0.5, -0.5, 0.0});
  footprint.push_back({0.5, -0.5, 0.0});
  footprint.push_back({0.5, 0.5, 0.0});
  footprint.push_back({-0.5, 0.5, 0.0});

  costmap.setFootprint(footprint);
  EXPECT_NEAR(costmap.getInscribedRadius(), 0.5, 0.01);
  EXPECT_NEAR(costmap.getCircumscribedRadius(), std::sqrt(0.5), 0.01);
}

TEST(LayeredCostmap, testUpdateMap)
{
  LayeredCostmap costmap("map", false, false);
  costmap.resizeMap(10, 10, 1.0, 0.0, 0.0);

  auto layer = std::make_shared<MockLayer>();
  costmap.addPlugin(layer);

  costmap.updateMap(5.0, 5.0, 0.0);

  EXPECT_TRUE(layer->update_bounds_called_);
  EXPECT_TRUE(layer->update_costs_called_);
  EXPECT_TRUE(costmap.isInitialized());
}

TEST(LayeredCostmap, testIsCurrent)
{
  LayeredCostmap costmap("map", false, false);
  auto layer = std::make_shared<MockLayer>();
  layer->setCurrent(true);
  costmap.addPlugin(layer);

  EXPECT_TRUE(costmap.isCurrent());
}

TEST(LayeredCostmap, testIsOutOfBounds)
{
  LayeredCostmap costmap("map", false, false);
  costmap.resizeMap(10, 10, 1.0, 0.0, 0.0);

  EXPECT_FALSE(costmap.isOutofBounds(5.0, 5.0));
  EXPECT_TRUE(costmap.isOutofBounds(15.0, 5.0));
}

TEST(LayeredCostmap, testGetBounds)
{
  LayeredCostmap costmap("map", false, false);
  costmap.resizeMap(10, 10, 1.0, 0.0, 0.0);

  auto layer = std::make_shared<MockLayer>();
  costmap.addPlugin(layer);

  costmap.updateMap(5.0, 5.0, 0.0);

  double minx, miny, maxx, maxy;
  costmap.getUpdatedBounds(minx, miny, maxx, maxy);
  // Bounds should be set from the layer
}

// ===================== LayerInterface Tests =====================

TEST(LayerInterface, testName)
{
  MockLayer layer;
  layer.setName("test_layer");
  EXPECT_EQ(layer.getName(), "test_layer");
}

TEST(LayerInterface, testCurrent)
{
  MockLayer layer;
  EXPECT_FALSE(layer.isCurrent());
  layer.setCurrent(true);
  EXPECT_TRUE(layer.isCurrent());
}

TEST(LayerInterface, testEnabled)
{
  MockLayer layer;
  EXPECT_TRUE(layer.isEnabled());
  layer.setEnabled(false);
  EXPECT_FALSE(layer.isEnabled());
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
