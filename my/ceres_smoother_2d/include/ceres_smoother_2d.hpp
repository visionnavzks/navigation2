#pragma once

/**
 * 基于 Ceres 的二维路径平滑器，带 ESDF 障碍物避让。
 *
 * 代价项：
 *   - 平滑度：二阶有限差分惩罚
 *   - 曲率：局部转角超过限制时的 hinge 惩罚（≤ min_turning_radius）
 *   - 参考路径：拉向 A* 参考路径的弹簧项
 *   - 长度：弹性带平方段长（均匀间距作用力）
 *   - 障碍物：基于 ESDF，将路径推离障碍物
 *
 * 所有梯度都通过 Ceres AutoDiff（Jet<double,N>）计算。无 ROS 依赖。
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

#include "ceres/ceres.h"

#include "esdf_map.hpp"
#include "smoother_params.hpp"

namespace ceres_smoother_2d
{

// 构造函数约定：
// 所有代价结构体在构造函数中接收预先计算好的 sqrt_w 并直接保存。
// 调用方负责只计算一次 sqrt(w)；结构体内部不能再调用 std::sqrt。

// 平滑度代价：惩罚二阶有限差分（离散加速度，而非 jerk）。
// jerk 会是 4 点三阶差分 p[i+2]-3*p[i+1]+3*p[i]-p[i-1]；
// 这里保留 3 点形式，因为它仍产生三对角 Hessian，且 Ceres AutoDiff 形式简单。
//   residual = sqrt_w * (p_next - 2*p_curr + p_prev)
struct SmoothnessCost
{
  explicit SmoothnessCost(double sqrt_w) : sqrt_w_(sqrt_w) {}

  template<typename T>
  bool operator()(const T * p_prev, const T * p_curr, const T * p_next, T * r) const
  {
    r[0] = sqrt_w_ * (p_next[0] - T(2.0) * p_curr[0] + p_prev[0]);
    r[1] = sqrt_w_ * (p_next[1] - T(2.0) * p_curr[1] + p_prev[1]);
    return true;
  }

  double sqrt_w_;
};

// 曲率代价：转角 hinge loss。
// 当实际转角 θ 超过允许上限 κ_max · ds 时惩罚，其中 ds 为局部平均步长。
//
// 与之前的点积亏损形式不同，这里直接用 atan2 计算角度，更直观：
//   θ = atan2(|cross|, dot)
//   violation = max(0, θ - κ_max · ds)
//
// 为兼容 Ceres AutoDiff，在 0 附近保持平滑，使用 sqrt(cross² + eps)
// 替代 abs(cross)。
struct CurvatureCost
{
  CurvatureCost(double sqrt_w, double max_kappa)
  : sqrt_w_(sqrt_w), max_kappa_(max_kappa) {}

  template<typename T>
  bool operator()(const T * p_prev, const T * p_curr, const T * p_next, T * r) const
  {
    const T v1x = p_curr[0] - p_prev[0];
    const T v1y = p_curr[1] - p_prev[1];
    const T v2x = p_next[0] - p_curr[0];
    const T v2y = p_next[1] - p_curr[1];

    const T n1 = ceres::sqrt(v1x * v1x + v1y * v1y + T(1e-12));
    const T n2 = ceres::sqrt(v2x * v2x + v2y * v2y + T(1e-12));
    const T ds = T(0.5) * (n1 + n2);

    const T dot = v1x * v2x + v1y * v2y;
    const T cross = v1x * v2y - v1y * v2x;

    // atan2(|cross|, dot) 得到 [0, π] 范围内的无符号转角。
    // 使用 sqrt(cross² + eps) 替代 abs(cross)，保证 0 附近平滑。
    const T theta = ceres::atan2(ceres::sqrt(cross * cross + T(1e-12)), dot);
    const T theta_limit = T(max_kappa_) * ds;
    const T violation = theta - theta_limit;

    // hinge 根据标量部分开关，模式与 ObstacleCostCeres 相同。
    double viol_scalar;
    if constexpr (std::is_same<T, double>::value) {
      viol_scalar = violation;
    } else {
      viol_scalar = violation.a;
    }
    r[0] = viol_scalar > 0.0 ? sqrt_w_ * violation : T(0.0);
    return true;
  }

  double sqrt_w_;
  double max_kappa_;
};

// 参考路径代价：惩罚偏离参考路径。
//   residual = sqrt_w * (p - p_ref)
struct ReferenceCost
{
  ReferenceCost(double x_ref, double y_ref, double sqrt_w)
  : x_ref_(x_ref), y_ref_(y_ref), sqrt_w_(sqrt_w) {}

  template<typename T>
  bool operator()(const T * p, T * r) const
  {
    r[0] = sqrt_w_ * (p[0] - T(x_ref_));
    r[1] = sqrt_w_ * (p[1] - T(y_ref_));
    return true;
  }

  double x_ref_, y_ref_;
  double sqrt_w_;
};

// 弹性带长度代价：最小化 Σ‖p_next - p_curr‖²（相邻点距离平方和）。
// 等价于 TEB / E-band 规划器中的“橡皮筋”力，没有静止长度，也没有 sqrt 非线性。
//
//   residual = sqrt_w * (p_next - p_curr)        // 2 个分量 (dx, dy)
//   Ceres 报告 0.5 * sum(residual²)，因此每段对 final_cost 的贡献为
//   0.5 * w * (dx² + dy²)。
//
// 相比 target_spacing 弹簧的优点：
//   - 纯线性残差 → 常量 Jacobian → Ceres 只需少量迭代即可收敛。
//     没有 sqrt(.) 非线性，也没有 1/||Δp|| 奇异点。
//   - 没有固定静止长度 → 不会与锁定的起终点冲突
//     （target_spacing × (N-1) 很少等于真实路径长度）。
//   - 当总长度受其他代价（平滑、参考、障碍）约束时，最小化 Σ‖Δs‖²
//     也会拉均匀段长，因此 resample_after_smooth 可以确定性地完成收尾。
struct PathLengthSquareCost
{
  explicit PathLengthSquareCost(double sqrt_w) : sqrt_w_(sqrt_w) {}

  template<typename T>
  bool operator()(const T * p_curr, const T * p_next, T * r) const
  {
    r[0] = sqrt_w_ * (p_next[0] - p_curr[0]);
    r[1] = sqrt_w_ * (p_next[1] - p_curr[1]);
    return true;
  }
  double sqrt_w_;
};

// 障碍物代价：两个项共同改善墙体附近行为。
//   residual[0] = sqrt_w_obstacle  * max(0, safe_dist - dist)   // 外侧 soft hinge
//   residual[1] = sqrt_w_penetrate * max(0, -dist)              // 内部强惩罚
//
// 第一项是标准对称 hinge：路径从外侧靠近安全边界时残差增大；
// 越过边界后（dist < 0）会进入平坦平台。第二项补上这个平台，
// 使点越深入障碍，优化器越持续向外推。例如位于墙内 0.3m 的点会支付
// `0.5 * w_penetration * 0.09`，而边界点只支付
// `0.5 * w_obstacle * safe_dist^2`。
//
// w_penetration=0 时可精确复现旧的单项行为。设置 w_penetration > 0 后，
// 障碍内部状态会严格劣于外部状态，从而消除仅靠 hinge 无法逃离的
// “卡在墙内”局部最小值。
//
// 查询使用双线性（不是双三次），原因见 ESDFMap::bilinearJet 的长注释。
// 双三次核会跨越障碍边界处的 ESDF 尖锐不连续并过冲，产生严重错误的距离
// （甚至梯度指向墙内）。双线性仅为 C^0，但始终受邻居 min/max 约束，
// 因此推离方向始终正确。
struct ObstacleCostCeres
{
  ObstacleCostCeres(
    const ESDFMap * map,
    double safe_dist,
    double sqrt_w_obstacle,
    double sqrt_w_penetrate)
  : map_(map),
    safe_dist_(safe_dist),
    sqrt_w_obstacle_(sqrt_w_obstacle),
    sqrt_w_penetrate_(sqrt_w_penetrate)
  {
  }

  template<typename T>
  bool operator()(const T * p, T * residual) const
  {
    // 带解析 Jet 导数的双线性 ESDF 查询。
    const T dist = map_->bilinearJet<T>(p[0], p[1]);
    const T diff = T(safe_dist_) - dist;
    // hinge 根据标量部分开关。Ceres Jet<T,N> 不能隐式转换为 T，
    // 因此需要查看 .a 成员；普通 double 会走恒等分支。
    double diff_scalar;
    if constexpr (std::is_same<T, double>::value) {
      diff_scalar = diff;
    } else {
      diff_scalar = diff.a;
    }
    residual[0] = diff_scalar > 0.0 ? sqrt_w_obstacle_ * diff : T(0.0);
    // 穿透代价：只有位于障碍内部时 -dist > 0。该项放在第二个残差上，
    // 使 AutoDiff 槽位数量与 AddResidualBlock<ObstacleCostCeres, 2, 2>
    // 中声明的一致。
    const T pen = -dist;
    double pen_scalar;
    if constexpr (std::is_same<T, double>::value) {
      pen_scalar = pen;
    } else {
      pen_scalar = pen.a;
    }
    residual[1] = pen_scalar > 0.0 ? sqrt_w_penetrate_ * pen : T(0.0);
    return true;
  }

  const ESDFMap * map_;
  double safe_dist_, sqrt_w_obstacle_, sqrt_w_penetrate_;
};

// ========================================================================
// 平滑结果
// ========================================================================
struct SmootherResult
{
  bool success{false};
  std::vector<double> x;
  std::vector<double> y;
  double final_cost{0.0};
  double solve_time_ms{0.0};
  int iterations{0};
  std::string report;
};

// ========================================================================
// resamplePathByArcLength
// ========================================================================
// 沿弧长均匀重采样折线，使相邻输出点约相隔 `target_spacing` 米。
// 精确保留首尾点（无端点漂移），并在每个输入线段内线性插值。
// smooth() 后可将其作为可选后处理步骤。
//
// 参数：
//   xs_in, ys_in：   输入折线（N >= 2）
//   target_spacing： 期望平均点间距（米）
//   xs_out, ys_out： 重采样后的输出折线
//
// 说明：
//   - 输出点数 M = max(2, round(L / target_spacing) + 1)，其中 L 为总弧长，
//     可保证平均间距 <= target。
//   - 若 N < 2、target_spacing <= 0 或总弧长近似为 0，则原样返回输入
//     （退化情况，无法重采样）。
inline void resamplePathByArcLength(
  const std::vector<double> & xs_in,
  const std::vector<double> & ys_in,
  double target_spacing,
  std::vector<double> & xs_out,
  std::vector<double> & ys_out)
{
  xs_out.clear();
  ys_out.clear();
  const int N = static_cast<int>(xs_in.size());
  if (N < 2 || target_spacing <= 0.0) {
    xs_out = xs_in;
    ys_out = ys_in;
    return;
  }

  // 每个输入顶点处的累计弧长。
  std::vector<double> cum(N, 0.0);
  for (int i = 1; i < N; ++i) {
    const double dx = xs_in[i] - xs_in[i - 1];
    const double dy = ys_in[i] - ys_in[i - 1];
    cum[i] = cum[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  const double total = cum.back();
  if (total < 1e-12) {
    // 所有输入点重合：无需重采样。
    xs_out = xs_in;
    ys_out = ys_in;
    return;
  }

  int M = static_cast<int>(std::round(total / target_spacing)) + 1;
  if (M < 2) {M = 2;}

  xs_out.resize(M);
  ys_out.resize(M);
  // 将端点锚定到原始顶点，精确保留起终点。
  xs_out[0] = xs_in.front();
  ys_out[0] = ys_in.front();
  xs_out[M - 1] = xs_in.back();
  ys_out[M - 1] = ys_in.back();

  // 对每个中间输出点，只遍历一次 cum[] 来寻找所在输入线段。
  // 输出弧长单调递增，因此线段索引只会向前移动：复杂度 O(N + M)，
  // 不必为每个点重新搜索。
  int i = 1;
  for (int j = 1; j < M - 1; ++j) {
    const double s = static_cast<double>(j) * total / static_cast<double>(M - 1);
    while (i < N - 1 && cum[i] < s) {++i;}
    const double seg_len = cum[i] - cum[i - 1];
    const double t = (seg_len > 1e-12) ? (s - cum[i - 1]) / seg_len : 0.0;
    xs_out[j] = xs_in[i - 1] + t * (xs_in[i] - xs_in[i - 1]);
    ys_out[j] = ys_in[i - 1] + t * (ys_in[i] - ys_in[i - 1]);
  }
}

// ========================================================================
// PathSmoother2D：基于 Ceres 的二维路径平滑器
// ========================================================================
class PathSmoother2D
{
public:
  explicit PathSmoother2D(SmootherParams params = {})
  : params_(std::move(params)) {}

  SmootherResult smooth(
    const std::vector<double> & x_in,
    const std::vector<double> & y_in,
    const ESDFMap & map) const
  {
    SmootherResult result;

    const int N_in = static_cast<int>(x_in.size());
    if (x_in.size() != y_in.size()) {
      result.success = false;
      result.report = "x_in and y_in size mismatch";
      return result;
    }
    if (N_in < 3) {
      result.x = x_in;
      result.y = y_in;
      result.success = true;
      return result;
    }

    // 前处理：可选地将输入重采样为均匀间距，使优化器从分布均匀的初值开始。
    // 如果不这样做，A* 风格输入中墙边密集、开阔区域稀疏的点分布会被继承，
    // 优化器需要额外对抗这种不均匀性。
    std::vector<double> xs = x_in;
    std::vector<double> ys = y_in;
    if (params_.resample_before_smooth && params_.resample_spacing > 0.0) {
      std::vector<double> rx, ry;
      resamplePathByArcLength(xs, ys, params_.resample_spacing, rx, ry);
      xs = std::move(rx);
      ys = std::move(ry);
    }
    const int N = static_cast<int>(xs.size());
    if (N < 3) {
      result.x = xs;
      result.y = ys;
      result.success = true;
      return result;
    }

    std::vector<std::array<double, 2>> path_optim(N);
    for (int i = 0; i < N; ++i) {
      path_optim[i] = {xs[i], ys[i]};
    }

    auto sqrt_weight = [](double w) {return std::sqrt(std::max(0.0, w));};

    const double sqrt_w_ref = sqrt_weight(params_.w_reference);
    const double sqrt_w_smooth = sqrt_weight(params_.w_smooth);
    const double sqrt_w_curv = sqrt_weight(params_.w_max_curvature);
    const double sqrt_w_length = sqrt_weight(params_.w_length);
    const double max_kappa = params_.maxCurvature();

    std::vector<double> obstacle_weight_stages;
    if (params_.w_obstacle <= 0.0) {
      obstacle_weight_stages.push_back(0.0);
    } else {
      double stage_weight = std::min(params_.w_obstacle, 2.0);
      obstacle_weight_stages.push_back(stage_weight);
      while (stage_weight * 10.0 < params_.w_obstacle) {
        stage_weight *= 10.0;
        obstacle_weight_stages.push_back(stage_weight);
      }
      if (obstacle_weight_stages.back() < params_.w_obstacle) {
        obstacle_weight_stages.push_back(params_.w_obstacle);
      }
    }

    ceres::Solver::Summary final_summary;
    double total_solve_time_ms = 0.0;
    int total_iterations = 0;
    bool all_stages_usable = true;
    std::ostringstream report;

    auto solve_stage = [&](double obstacle_weight) {
        ceres::Problem problem;
        for (int i = 0; i < N; ++i) {
          problem.AddParameterBlock(path_optim[i].data(), 2);
        }
        problem.SetParameterBlockConstant(path_optim[0].data());
        problem.SetParameterBlockConstant(path_optim[N - 1].data());

        const double sqrt_w_obs = sqrt_weight(obstacle_weight);
        // 穿透代价使用独立（解耦）的 sqrt 权重，使用户可以独立调节
        // “软障碍”和“硬墙”行为。它不会随 obstacle_weight_stages 缩放：
        // 分阶段提升的目标是先用较低 w_obstacle 找到合理形状，再逐步收紧。
        // 穿透项应始终全强度开启；即使在 stage 0，也不希望优化器停在墙内。
        const double sqrt_w_pen = sqrt_weight(params_.w_penetration);
        // 障碍物代价直接使用 ESDFMap 支持 Jet 的双线性查询。
        // 为什么使用双线性而非双三次，见 ObstacleCostCeres 注释。
        for (int i = 0; i < N; ++i) {
          // 中间节点代价（位置 + 障碍物）。
          if (i > 0 && i < N - 1) {
            if (params_.w_reference > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ReferenceCost, 2, 2>(
                  new ReferenceCost(xs[i], ys[i], sqrt_w_ref)),
                nullptr, path_optim[i].data());
            }
            if (obstacle_weight > 0.0 || sqrt_w_pen > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ObstacleCostCeres, 2, 2>(
                  new ObstacleCostCeres(
                    &map, params_.obstacleCostDistance(), sqrt_w_obs, sqrt_w_pen)),
                nullptr, path_optim[i].data());
            }
          }

          // 弹性带长度代价：相邻点距离平方。
          // 2 个残差 (dx, dy) -> 常量 Jacobian -> 快速收敛。
          if (params_.w_length > 0.0 && i < N - 1) {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<PathLengthSquareCost, 2, 2, 2>(
                new PathLengthSquareCost(sqrt_w_length)),
              nullptr, path_optim[i].data(), path_optim[i + 1].data());
          }

          // 三点几何约束（i = 1 .. N-2）。
          if (i > 0 && i < N - 1) {
            if (params_.w_smooth > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SmoothnessCost, 2, 2, 2, 2>(
                  new SmoothnessCost(sqrt_w_smooth)),
                nullptr,
                path_optim[i - 1].data(), path_optim[i].data(), path_optim[i + 1].data());
            }
            if (params_.w_max_curvature > 0.0) {
              problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CurvatureCost, 1, 2, 2, 2>(
                  new CurvatureCost(sqrt_w_curv, max_kappa)),
                nullptr,
                path_optim[i - 1].data(), path_optim[i].data(), path_optim[i + 1].data());
            }
          }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = params_.max_iterations;
        options.max_solver_time_in_seconds = params_.max_time_seconds;
        options.minimizer_progress_to_stdout = params_.verbose;
        options.logging_type = params_.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
        // 对少于约 2k 变量的问题，线程开销超过并行带来的收益。
        options.num_threads = 1;

        ceres::Solver::Summary summary;
        auto t0 = std::chrono::steady_clock::now();
        ceres::Solve(options, &problem, &summary);
        auto t1 = std::chrono::steady_clock::now();

        total_solve_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_iterations += static_cast<int>(summary.iterations.size());
        final_summary = summary;
        all_stages_usable = all_stages_usable && summary.IsSolutionUsable();

        double min_dist = std::numeric_limits<double>::infinity();
        double min_margin = std::numeric_limits<double>::infinity();
        int active_count = 0;
        const double obstacle_threshold = params_.obstacleCostDistance();
        for (int i = 1; i < N - 1; ++i) {
          const double d = map.bilinearJet<double>(path_optim[i][0], path_optim[i][1]);
          const double margin = d - obstacle_threshold;
          min_dist = std::min(min_dist, d);
          min_margin = std::min(min_margin, margin);
          if (d < obstacle_threshold) {
            ++active_count;
          }
        }

        std::ostringstream stage_report;
        stage_report << "stage w_obstacle=" << obstacle_weight
                     << ", final_cost=" << summary.final_cost
                     << ", min_d=" << min_dist
                     << ", min_margin=" << min_margin
                     << ", active_count=" << active_count
                     << ", " << summary.BriefReport() << '\n';
        report << stage_report.str();
        if (params_.verbose) {
          std::cout << stage_report.str();
        }
      };

    for (double obstacle_weight : obstacle_weight_stages) {
      solve_stage(obstacle_weight);
      if (!all_stages_usable) {
        break;
      }
    }

    result.solve_time_ms = total_solve_time_ms;
    result.final_cost = final_summary.final_cost;
    result.iterations = total_iterations;
    result.report = report.str();
    result.success = all_stages_usable && final_summary.IsSolutionUsable();

    result.x.resize(N);
    result.y.resize(N);
    for (int i = 0; i < N; ++i) {
      result.x[i] = path_optim[i][0];
      result.y[i] = path_optim[i][1];
    }

    // 可选后处理：沿平滑路径强制均匀点间距。启用后，结果点数可能多于
    // 或少于输入；起终点位置保持不变。
    if (params_.resample_after_smooth && params_.resample_spacing > 0.0) {
      std::vector<double> rx, ry;
      resamplePathByArcLength(result.x, result.y, params_.resample_spacing, rx, ry);
      result.x = std::move(rx);
      result.y = std::move(ry);
    }

    return result;
  }

private:
  SmootherParams params_;
};

}  // namespace ceres_smoother_2d
