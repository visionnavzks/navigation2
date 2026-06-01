#ifndef SMAC_PLANNER__COSTMAP_DOWNSAMPLER_HPP_
#define SMAC_PLANNER__COSTMAP_DOWNSAMPLER_HPP_

#include <memory>
#include <string>
#include <cmath>

#include "my/smac_planner/costmap_2d.hpp"
#include "my/smac_planner/constants.hpp"

namespace smac_planner
{

class CostmapDownsampler
{
public:
  CostmapDownsampler();
  ~CostmapDownsampler();

  void on_configure(
    Costmap2D * const costmap,
    const unsigned int & downsampling_factor,
    const bool & use_min_cost_neighbor = false);

  Costmap2D * downsample(const unsigned int & downsampling_factor);

  void resizeCostmap();

protected:
  void updateCostmapSize();
  void setCostOfCell(const unsigned int & new_mx, const unsigned int & new_my);

  unsigned int _size_x, _size_y;
  unsigned int _downsampled_size_x, _downsampled_size_y;
  unsigned int _downsampling_factor;
  bool _use_min_cost_neighbor;
  float _downsampled_resolution;
  Costmap2D * _costmap;
  std::unique_ptr<Costmap2D> _downsampled_costmap;
};

}  // namespace smac_planner

#endif  // SMAC_PLANNER__COSTMAP_DOWNSAMPLER_HPP_
