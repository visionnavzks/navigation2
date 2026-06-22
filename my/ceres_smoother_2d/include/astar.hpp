#pragma once

/**
 * 在二维占据栅格上执行带机器人半径膨胀的 A* 最短路径搜索。
 *
 * 用作 Web demo 中较慢 Python A* 的直接替代实现。在 1436x847 的
 * occupancy_map.png 上通常可提速 50-200 倍。
 *
 * 算法：
 *   - 8 邻接栅格，欧氏步进代价（直连为 1，对角为 sqrt(2)）
 *   - 欧氏距离启发式（对 8 邻接 admissible 且 consistent）
 *   - 二叉堆 + 惰性删除（不显式 decrease-key）
 *   - g_score / came_from / closed 使用扁平数组，便于缓存访问
 *
 * 障碍物膨胀：
 *   当 robot_radius > 0 时，会先预处理栅格：ESDF 距离 < robot_radius
 *   的任意单元都视为障碍。得到的路径对该半径的圆形机器人可行。
 *   这是 TEB / navfn / ROS 规划器中的常见做法。
 *
 * 输入/输出均使用世界坐标（米）。内部算法在栅格单元上运行；坐标转换
 * 与 Python 参考实现一致（int 向 0 截断）。
 *
 * 边界情况：
 *   - 起点或终点位于（膨胀后）障碍内 → success=false，路径为空
 *   - 起点 == 终点 → success=true，单点路径
 *   - 不存在路径 → success=false，路径为空
 *   - 世界坐标越界 → clamp 到最近的栅格单元
 *
 * 无 ROS 依赖。头文件实现，仅依赖 C++ 标准库。
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

#include "esdf_map.hpp"

namespace ceres_smoother_2d
{

struct AStarResult
{
  bool success{false};
  std::vector<double> x;       // 世界坐标 x（米）
  std::vector<double> y;       // 世界坐标 y（米）
  int expansions{0};           // 出队并展开的节点数
  double time_ms{0.0};         // 搜索耗时（墙钟时间）
};

// 8 邻接偏移和步进代价。
namespace astar_detail
{
// 直连 = 1，对角 = sqrt(2)。顺序：W, E, S, N, NW, NE, SW, SE。
// 顺序不影响结果，按类别分组是为了让内层循环更友好地访问缓存。
constexpr int kNumDirs = 8;
constexpr int kNumCardinalDirs = 4;  // 前 4 个为正交方向（W,E,S,N），其余为对角。
constexpr int kDx[kNumDirs] = {-1, 1, 0, 0, -1, 1, -1, 1};
constexpr int kDy[kNumDirs] = { 0, 0,-1, 1, -1,-1,  1, 1};
constexpr double kDiag = 1.4142135623730951;
constexpr double kDc[kNumDirs] = {
  1.0, 1.0, 1.0, 1.0,
  kDiag, kDiag, kDiag, kDiag};

struct Cell
{
  int x{0};
  int y{0};
};

struct OpenNode
{
  double f{0.0};
  int32_t idx{-1};
  bool operator>(const OpenNode & o) const {return f > o.f;}
};

inline int clampIndex(int v, int upper_exclusive)
{
  return std::max(0, std::min(upper_exclusive - 1, v));
}

inline int32_t toIndex(int x, int y, int width)
{
  return static_cast<int32_t>(y) * width + x;
}

inline Cell fromIndex(int32_t idx, int width)
{
  return {idx % width, idx / width};
}

inline double heuristic(const Cell & a, const Cell & b)
{
  const double dx = static_cast<double>(std::abs(a.x - b.x));
  const double dy = static_cast<double>(std::abs(a.y - b.y));
  const double mn = std::min(dx, dy);
  const double mx = std::max(dx, dy);
  // 空 8 邻接栅格上的精确最短路距离，直连代价为 1，对角代价为 sqrt(2)。
  // 它比欧氏距离更紧，同时仍保持 admissible 和 consistent。
  return (mx - mn) + kDiag * mn;
}

inline Cell worldToCell(
  const ESDFMap & map,
  double wx,
  double wy)
{
  // 有意使用向 0 截断，以匹配原 Python 版
  // `int((world - origin) / resolution)` 的行为。
  const int x = static_cast<int>((wx - map.originX()) / map.resolution());
  const int y = static_cast<int>((wy - map.originY()) / map.resolution());
  return {
    clampIndex(x, map.width()),
    clampIndex(y, map.height())};
}

inline double cellToWorldX(const ESDFMap & map, int x)
{
  return (static_cast<double>(x) + 0.5) * map.resolution() + map.originX();
}

inline double cellToWorldY(const ESDFMap & map, int y)
{
  return (static_cast<double>(y) + 0.5) * map.resolution() + map.originY();
}

inline void appendPathPoint(
  const ESDFMap & map,
  int32_t idx,
  AStarResult & res)
{
  const Cell c = fromIndex(idx, map.width());
  res.x.push_back(cellToWorldX(map, c.x));
  res.y.push_back(cellToWorldY(map, c.y));
}

inline void reconstructPath(
  const ESDFMap & map,
  int32_t start_idx,
  int32_t goal_idx,
  const std::vector<int32_t> & came_from,
  int reserve_hint,
  AStarResult & res)
{
  std::vector<int32_t> rev;
  rev.reserve(static_cast<size_t>(std::max(0, reserve_hint)) + 1);
  for (int32_t cur = goal_idx; cur != -1; cur = came_from[cur]) {
    rev.push_back(cur);
    if (cur == start_idx) {break;}
  }
  std::reverse(rev.begin(), rev.end());

  res.x.reserve(rev.size());
  res.y.reserve(rev.size());
  for (int32_t idx : rev) {
    appendPathPoint(map, idx, res);
  }
}
}  // namespace astar_detail

inline AStarResult astarSolve(
  const ESDFMap & map,
  double sx_w, double sy_w,
  double gx_w, double gy_w,
  double robot_radius = 0.0)
{
  AStarResult res;

  const int W = map.width();
  const int H = map.height();
  if (W <= 0 || H <= 0) {return res;}
  const size_t N = static_cast<size_t>(W) * static_cast<size_t>(H);

  // 「能否通行」直接由 ESDF 判断，无需预先物化一张膨胀栅格：
  //   robot_radius > 0 → 离最近障碍 < robot_radius 的格子视为障碍（膨胀），
  //                      使中心线路径对该半径的圆形机器人可行；
  //   robot_radius <= 0 → 用原始占据栅格（点机器人）。
  const auto & occ = map.occupancyGrid();
  const auto & esdf = map.esdfGrid();
  const bool inflate = robot_radius > 0.0;
  auto blocked = [&](int32_t idx) {
      const size_t i = static_cast<size_t>(idx);
      return inflate ? (esdf[i] < robot_radius) : (occ[i] != 0);
    };

  const astar_detail::Cell start = astar_detail::worldToCell(map, sx_w, sy_w);
  const astar_detail::Cell goal = astar_detail::worldToCell(map, gx_w, gy_w);
  const int32_t start_idx = astar_detail::toIndex(start.x, start.y, W);
  const int32_t goal_idx = astar_detail::toIndex(goal.x, goal.y, W);

  // 起点或终点在（膨胀后）障碍内时直接拒绝。
  if (blocked(start_idx) || blocked(goal_idx)) {
    return res;  // success=false
  }

  // 平凡情况：起点即终点。
  if (start_idx == goal_idx) {
    res.success = true;
    astar_detail::appendPathPoint(map, start_idx, res);
    return res;
  }

  // 扁平数组更利于缓存，相比 unordered_map 通常快约 10 倍。
  std::vector<double> g_score(N, std::numeric_limits<double>::infinity());
  std::vector<int32_t> came_from(N, -1);  // 父栅格索引，或 -1
  std::vector<uint8_t> closed(N, 0);      // 0/1 标记

  // open 集：按 f-score 排序的小根堆。弹出时通过 closed[] 做惰性删除。
  std::priority_queue<
    astar_detail::OpenNode,
    std::vector<astar_detail::OpenNode>,
    std::greater<astar_detail::OpenNode>> open;

  g_score[start_idx] = 0.0;
  open.push({astar_detail::heuristic(start, goal), start_idx});

  const auto t0 = std::chrono::steady_clock::now();

  while (!open.empty()) {
    const astar_detail::OpenNode cur = open.top();
    open.pop();
    const int32_t cur_idx = cur.idx;

    // 跳过过期项：之后已经压入过更好的 f-score。
    if (closed[cur_idx]) {continue;}

    // 标记为最终节点：此时该节点的 g_score 已经最优。
    closed[cur_idx] = 1;
    ++res.expansions;

    // 终点被弹出后停止展开，此时终点 g_score 已经最优。
    if (cur_idx == goal_idx) {break;}

    const astar_detail::Cell cur_cell = astar_detail::fromIndex(cur_idx, W);
    const double g_cur = g_score[cur_idx];

    for (int k = 0; k < astar_detail::kNumDirs; ++k) {
      const int nx = cur_cell.x + astar_detail::kDx[k];
      const int ny = cur_cell.y + astar_detail::kDy[k];
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) {continue;}
      // 对角移动禁止穿角：要求两个正交相邻格都空闲，否则路径会从两个
      // 对角相接的障碍之间「贴角」穿过（robot_radius=0 时尤为明显）。
      // 两个正交格 (nx,cur.y)、(cur.x,ny) 必在界内，无需再做边界检查。
      if (k >= astar_detail::kNumCardinalDirs &&
        (blocked(astar_detail::toIndex(nx, cur_cell.y, W)) ||
        blocked(astar_detail::toIndex(cur_cell.x, ny, W))))
      {
        continue;
      }
      const astar_detail::Cell next{nx, ny};
      const int32_t nidx = astar_detail::toIndex(next.x, next.y, W);
      if (blocked(nidx) || closed[nidx]) {continue;}

      const double tentative_g = g_cur + astar_detail::kDc[k];
      if (tentative_g >= g_score[nidx]) {continue;}

      g_score[nidx] = tentative_g;
      came_from[nidx] = cur_idx;
      open.push({tentative_g + astar_detail::heuristic(next, goal), nidx});
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // 只有终点被最终确定（g_score 有限）才算成功。
  if (!std::isfinite(g_score[goal_idx])) {return res;}  // success=false

  astar_detail::reconstructPath(map, start_idx, goal_idx, came_from, res.expansions, res);
  res.success = true;
  return res;
}

}  // namespace ceres_smoother_2d
