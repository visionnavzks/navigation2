#include "my/hybrid_astar/costmap_downsampler.hpp"

#include <algorithm>

namespace hybrid_astar
{

CostmapDownsampler::CostmapDownsampler()
: _costmap(nullptr),
  _downsampled_costmap(nullptr)
{
}

CostmapDownsampler::~CostmapDownsampler()
{
}

void CostmapDownsampler::on_configure(
  Costmap2D * const costmap,
  const unsigned int & downsampling_factor,
  const bool & use_min_cost_neighbor)
{
  _costmap = costmap;
  _downsampling_factor = downsampling_factor;
  _use_min_cost_neighbor = use_min_cost_neighbor;
  updateCostmapSize();

  _downsampled_costmap = std::make_unique<Costmap2D>(
    _downsampled_size_x, _downsampled_size_y, _downsampled_resolution,
    _costmap->getOriginX(), _costmap->getOriginY(),
    static_cast<unsigned char>(UNKNOWN_COST));
}

Costmap2D * CostmapDownsampler::downsample(const unsigned int & downsampling_factor)
{
  _downsampling_factor = downsampling_factor;
  updateCostmapSize();

  if (_downsampled_costmap->getSizeInCellsX() != _downsampled_size_x ||
    _downsampled_costmap->getSizeInCellsY() != _downsampled_size_y ||
    _downsampled_costmap->getResolution() != _downsampled_resolution)
  {
    resizeCostmap();
  }

  for (unsigned int i = 0; i < _downsampled_size_x; ++i) {
    for (unsigned int j = 0; j < _downsampled_size_y; ++j) {
      setCostOfCell(i, j);
    }
  }

  return _downsampled_costmap.get();
}

void CostmapDownsampler::updateCostmapSize()
{
  _size_x = _costmap->getSizeInCellsX();
  _size_y = _costmap->getSizeInCellsY();
  _downsampled_size_x = static_cast<unsigned int>(
    ceil(static_cast<float>(_size_x) / static_cast<float>(_downsampling_factor)));
  _downsampled_size_y = static_cast<unsigned int>(
    ceil(static_cast<float>(_size_y) / static_cast<float>(_downsampling_factor)));
  _downsampled_resolution = static_cast<float>(_downsampling_factor) * _costmap->getResolution();
}

void CostmapDownsampler::resizeCostmap()
{
  _downsampled_costmap->resizeMap(
    _downsampled_size_x,
    _downsampled_size_y,
    _downsampled_resolution,
    _costmap->getOriginX(),
    _costmap->getOriginY());
}

void CostmapDownsampler::setCostOfCell(
  const unsigned int & new_mx,
  const unsigned int & new_my)
{
  unsigned int mx, my;
  unsigned char cost = _use_min_cost_neighbor ? 255 : 0;
  unsigned int x_offset = new_mx * _downsampling_factor;
  unsigned int y_offset = new_my * _downsampling_factor;

  for (unsigned int i = 0; i < _downsampling_factor; ++i) {
    mx = x_offset + i;
    if (mx >= _size_x) {
      continue;
    }
    for (unsigned int j = 0; j < _downsampling_factor; ++j) {
      my = y_offset + j;
      if (my >= _size_y) {
        continue;
      }
      cost = _use_min_cost_neighbor ?
        std::min(cost, _costmap->getCost(mx, my)) :
        std::max(cost, _costmap->getCost(mx, my));
    }
  }

  _downsampled_costmap->setCost(new_mx, new_my, cost);
}

}  // namespace hybrid_astar
