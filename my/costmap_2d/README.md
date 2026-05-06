# Standalone Costmap 2D

This folder contains a small ROS-free extraction of the `nav2_costmap_2d` grid and filter logic.
It keeps the data model and algorithms needed for local experiments while replacing ROS messages,
TF, services, subscriptions, and lifecycle hooks with plain C++ structures.

Included pieces:

- `Costmap2D`: row-major `unsigned char` cost grid, coordinate conversion, window copying, origin updates, polygon filling, and PGM export.
- `OccupancyGrid`: lightweight replacement for `nav_msgs::msg::OccupancyGrid` metadata and `int8_t` data.
- Cost constants and occupancy-grid conversion helpers.
- Layer merge helpers matching the Nav2 overwrite, max, max-without-unknown-overwrite, and addition policies.
- `KeepoutFilter`: standalone processing of a mask over master-grid update windows, including the standard overlapping-window scenario.

Build and test:

```bash
cmake -S my/costmap_2d -B my/costmap_2d/build
cmake --build my/costmap_2d/build
ctest --test-dir my/costmap_2d/build --output-on-failure
```
