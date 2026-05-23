#ifndef COSTMAP_2D__COSTMAP_FILTER_HPP_
#define COSTMAP_2D__COSTMAP_FILTER_HPP_

#include "costmap_2d.hpp"

namespace costmap_2d {

class CostmapFilter {
public:
  virtual ~CostmapFilter() = default;

  void setEnabled(bool enabled) { enabled_ = enabled; }
  bool isEnabled() const { return enabled_; }

  virtual void process(Costmap2D &master_grid, int min_i, int min_j, int max_i,
                       int max_j) = 0;

  int8_t getMaskData(const OccupancyGrid &filter_mask, unsigned int mx,
                     unsigned int my) const {
    return filter_mask.getData(mx, my);
  }

  unsigned char getMaskCost(const OccupancyGrid &filter_mask, unsigned int mx,
                            unsigned int my) const {
    return occupancyToCost(getMaskData(filter_mask, mx, my));
  }

protected:
  bool enabled_{true};
};

} // namespace costmap_2d

#endif // COSTMAP_2D__COSTMAP_FILTER_HPP_