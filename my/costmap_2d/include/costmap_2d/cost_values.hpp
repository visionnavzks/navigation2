#ifndef COSTMAP_2D__COST_VALUES_HPP_
#define COSTMAP_2D__COST_VALUES_HPP_

#include <cstdint>

namespace costmap_2d {

enum class CombinationMethod : int {
  Overwrite = 0,
  Max = 1,
  MaxWithoutUnknownOverwrite = 2,
  Addition = 3
};

static constexpr unsigned char NO_INFORMATION = 255;
static constexpr unsigned char LETHAL_OBSTACLE = 254;
static constexpr unsigned char INSCRIBED_INFLATED_OBSTACLE = 253;
static constexpr unsigned char MAX_NON_OBSTACLE = 252;
static constexpr unsigned char FREE_SPACE = 0;

static constexpr int8_t OCC_GRID_UNKNOWN = -1;
static constexpr int8_t OCC_GRID_FREE = 0;
static constexpr int8_t OCC_GRID_OCCUPIED = 100;

static constexpr uint8_t KEEPOUT_FILTER = 0;
static constexpr uint8_t SPEED_FILTER_PERCENT = 1;
static constexpr uint8_t SPEED_FILTER_ABSOLUTE = 2;
static constexpr uint8_t BINARY_FILTER = 3;

static constexpr double BASE_DEFAULT = 0.0;
static constexpr double MULTIPLIER_DEFAULT = 1.0;

static constexpr int8_t SPEED_MASK_UNKNOWN = -1;
static constexpr int8_t SPEED_MASK_NO_LIMIT = 0;
static constexpr double NO_SPEED_LIMIT = 0.0;

} // namespace costmap_2d

#endif // COSTMAP_2D__COST_VALUES_HPP_