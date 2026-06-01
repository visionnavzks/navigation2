// Costmap Layer Algorithms Tests

#include <gtest/gtest.h>
#include <vector>

#include "my_costmap_2d/costmap_layer_algorithms.hpp"
#include "my_costmap_2d/costmap_2d.hpp"
#include "my_costmap_2d/cost_values.hpp"

using namespace my_costmap_2d;

// ===================== UpdateWithTrueOverwrite Tests =====================

TEST(LayerAlgorithms, updateWithTrueOverwrite)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  std::vector<unsigned char> layer_data(100, LETHAL_OBSTACLE);

  updateWithTrueOverwrite(layer_data.data(), master, 0, 0, 10, 10);

  for (unsigned int i = 0; i < 10; ++i) {
    for (unsigned int j = 0; j < 10; ++j) {
      EXPECT_EQ(master.getCost(i, j), LETHAL_OBSTACLE);
    }
  }
}

TEST(LayerAlgorithms, updateWithTrueOverwritePartial)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);

  updateWithTrueOverwrite(layer_data.data(), master, 2, 2, 5, 5);

  for (unsigned int i = 0; i < 10; ++i) {
    for (unsigned int j = 0; j < 10; ++j) {
      if (i >= 2 && i < 5 && j >= 2 && j < 5) {
        EXPECT_EQ(master.getCost(i, j), FREE_SPACE);
      } else {
        EXPECT_EQ(master.getCost(i, j), FREE_SPACE);
      }
    }
  }
}

// ===================== UpdateWithOverwrite Tests =====================

TEST(LayerAlgorithms, updateWithOverwrite)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  std::vector<unsigned char> layer_data(100, NO_INFORMATION);
  layer_data[55] = LETHAL_OBSTACLE;

  updateWithOverwrite(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), LETHAL_OBSTACLE);
}

TEST(LayerAlgorithms, updateWithOverwriteSkipsNoInfo)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 200);
  std::vector<unsigned char> layer_data(100, NO_INFORMATION);

  updateWithOverwrite(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 200u);
}

// ===================== UpdateWithMax Tests =====================

TEST(LayerAlgorithms, updateWithMax)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 200;

  updateWithMax(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 200u);
}

TEST(LayerAlgorithms, updateWithMaxExistingHigher)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 200);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 100;

  updateWithMax(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 200u);
}

TEST(LayerAlgorithms, updateWithMaxMasterUnknown)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0, NO_INFORMATION);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 100;

  updateWithMax(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 100u);
}

TEST(LayerAlgorithms, updateWithMaxLayerUnknown)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 100);
  std::vector<unsigned char> layer_data(100, NO_INFORMATION);

  updateWithMax(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 100u);
}

// ===================== UpdateWithMaxWithoutUnknownOverwrite Tests =====================

TEST(LayerAlgorithms, updateWithMaxWithoutUnknownOverwrite)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 100);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 200;

  updateWithMaxWithoutUnknownOverwrite(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 200u);
}

TEST(LayerAlgorithms, updateWithMaxWithoutUnknownOverwriteMasterUnknown)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0, NO_INFORMATION);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 200;

  updateWithMaxWithoutUnknownOverwrite(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), static_cast<unsigned char>(NO_INFORMATION));
}

// ===================== UpdateWithAddition Tests =====================

TEST(LayerAlgorithms, updateWithAddition)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 50);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 30;

  updateWithAddition(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 80u);
}

TEST(LayerAlgorithms, updateWithAdditionClamp)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 200);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 200;

  updateWithAddition(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), static_cast<unsigned char>(INSCRIBED_INFLATED_OBSTACLE - 1));
}

TEST(LayerAlgorithms, updateWithAdditionMasterUnknown)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0, NO_INFORMATION);
  std::vector<unsigned char> layer_data(100, FREE_SPACE);
  layer_data[55] = 100;

  updateWithAddition(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 100u);
}

TEST(LayerAlgorithms, updateWithAdditionLayerUnknown)
{
  Costmap2D master(10, 10, 1.0, 0.0, 0.0);
  master.setCost(5, 5, 100);
  std::vector<unsigned char> layer_data(100, NO_INFORMATION);

  updateWithAddition(layer_data.data(), master, 0, 0, 10, 10);

  EXPECT_EQ(master.getCost(5, 5), 100u);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
