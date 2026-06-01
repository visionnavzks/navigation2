#pragma once

#include <string>
#include <vector>

namespace nonline_smoother
{

struct SmootherParams
{
  double max_kappa{0.5};
  double w_ref{10.0};
  double w_dkappa{10.0};
  double w_kappa{0.1};
  double w_ds{1.0};
  double target_ds{0.0};
  bool has_kappa_start{true};
  double kappa_start{0.0};
  double ds_min_ratio{0.05};
  double ds_max_ratio{2.0};
  double dynamic_weight{100.0};
  int max_num_iterations{100};
  int num_threads{1};
  bool verbose{false};
};

struct SmootherResult
{
  bool success{false};
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> theta;
  std::vector<double> kappa;
  std::vector<double> ds;
  std::vector<double> dkappa;
  std::vector<double> gears;
  std::vector<bool> is_virtual;
  double target_ds{0.0};
  double solve_time_ms{0.0};
  double final_cost{0.0};
  int iterations{0};
  std::string brief_report;
};

class CeresPathSmoother
{
public:
  explicit CeresPathSmoother(SmootherParams params = {});

  SmootherResult solve(
    const std::vector<double> & x_ref,
    const std::vector<double> & y_ref,
    const std::vector<double> & theta_ref,
    const std::vector<double> & gears) const;

private:
  SmootherParams params_;
};

}  // namespace nonline_smoother