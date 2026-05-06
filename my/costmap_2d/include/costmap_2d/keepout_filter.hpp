#ifndef COSTMAP_2D__KEEPOUT_FILTER_HPP_
#define COSTMAP_2D__KEEPOUT_FILTER_HPP_

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "costmap_filter.hpp"

namespace costmap_2d {

class KeepoutFilter : public CostmapFilter {
public:
  void setMask(const OccupancyGrid &mask) {
    if (!mask.valid()) {
      throw std::invalid_argument("KeepoutFilter mask is empty or malformed");
    }
    filter_mask_ = mask;
    has_filter_mask_ = true;
  }

  void setGlobalFrame(std::string frame_id) {
    global_frame_ = std::move(frame_id);
  }

  void setGlobalToMaskTransform(const Transform2D &transform) {
    global_to_mask_ = transform;
    has_global_to_mask_ = true;
  }

  void clearGlobalToMaskTransform() { has_global_to_mask_ = false; }

  void setLethalOverride(bool enabled, unsigned char cost = MAX_NON_OBSTACLE) {
    override_lethal_cost_ = enabled;
    lethal_override_cost_ = static_cast<unsigned char>(std::min<unsigned int>(
        std::max<unsigned int>(cost, FREE_SPACE), MAX_NON_OBSTACLE));
  }

  void setPoseLethal(bool lethal) { is_pose_lethal_ = lethal; }

  bool isActive() const { return has_filter_mask_; }

  void process(Costmap2D &master_grid, int min_i, int min_j, int max_i,
               int max_j) override {
    if (!enabled_ || !has_filter_mask_) {
      return;
    }

    int mg_min_x = min_i;
    int mg_min_y = min_j;
    int mg_max_x = max_i;
    int mg_max_y = max_j;
    const OccupancyGrid &mask = filter_mask_;
    const bool same_frame = mask.frame_id == global_frame_;
    if (!same_frame && !has_global_to_mask_) {
      return;
    }

    if (same_frame) {
      double wx, wy;
      const double half_cell_size = 0.5 * mask.resolution;
      wx = mask.origin_x + half_cell_size;
      wy = mask.origin_y + half_cell_size;
      master_grid.worldToMapNoBounds(wx, wy, mg_min_x, mg_min_y);
      if (mg_min_x >= max_i || mg_min_y >= max_j) {
        return;
      }
      mg_min_x = std::max(min_i, mg_min_x);
      mg_min_y = std::max(min_j, mg_min_y);

      wx = mask.origin_x + mask.width * mask.resolution + half_cell_size;
      wy = mask.origin_y + mask.height * mask.resolution + half_cell_size;
      master_grid.worldToMapNoBounds(wx, wy, mg_max_x, mg_max_y);
      if (mg_max_x <= min_i || mg_max_y <= min_j) {
        return;
      }
      mg_max_x = std::min(max_i, mg_max_x);
      mg_max_y = std::min(max_j, mg_max_y);
    }

    mg_min_x = std::min(std::max(mg_min_x, 0),
                        static_cast<int>(master_grid.getSizeInCellsX()));
    mg_min_y = std::min(std::max(mg_min_y, 0),
                        static_cast<int>(master_grid.getSizeInCellsY()));
    mg_max_x = std::min(std::max(mg_max_x, 0),
                        static_cast<int>(master_grid.getSizeInCellsX()));
    mg_max_y = std::min(std::max(mg_max_y, 0),
                        static_cast<int>(master_grid.getSizeInCellsY()));
    if (mg_min_x >= mg_max_x || mg_min_y >= mg_max_y) {
      return;
    }

    unsigned char *master_array = master_grid.getCharMap();
    for (unsigned int i = static_cast<unsigned int>(mg_min_x);
         i < static_cast<unsigned int>(mg_max_x); ++i) {
      for (unsigned int j = static_cast<unsigned int>(mg_min_y);
           j < static_cast<unsigned int>(mg_max_y); ++j) {
        const unsigned int index = master_grid.getIndex(i, j);
        const unsigned char old_data = master_array[index];
        double global_wx, global_wy;
        master_grid.mapToWorld(i, j, global_wx, global_wy);

        double mask_wx = global_wx;
        double mask_wy = global_wy;
        if (!same_frame && has_global_to_mask_) {
          const Point transformed =
              global_to_mask_.apply(Point{global_wx, global_wy, 0.0});
          mask_wx = transformed.x;
          mask_wy = transformed.y;
        }

        unsigned int mx, my;
        if (!worldToMap(mask, mask_wx, mask_wy, mx, my)) {
          continue;
        }

        const unsigned char data = getMaskCost(mask, mx, my);
        if (data == NO_INFORMATION) {
          continue;
        }
        if (data > old_data || old_data == NO_INFORMATION) {
          master_array[index] = override_lethal_cost_ && is_pose_lethal_
                                    ? lethal_override_cost_
                                    : data;
        }
      }
    }

    last_pose_lethal_ = is_pose_lethal_;
  }

private:
  OccupancyGrid filter_mask_;
  bool has_filter_mask_{false};
  std::string global_frame_{"map"};
  Transform2D global_to_mask_;
  bool has_global_to_mask_{false};
  bool override_lethal_cost_{false};
  unsigned char lethal_override_cost_{MAX_NON_OBSTACLE};
  bool last_pose_lethal_{false};
  bool is_pose_lethal_{false};
};

} // namespace costmap_2d

#endif // COSTMAP_2D__KEEPOUT_FILTER_HPP_