#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ceres_nonlinear_smoother.hpp"

namespace
{

std::vector<double> linspace(double start, double end, int count)
{
  std::vector<double> values;
  values.reserve(count);
  if (count == 1) {
    values.push_back(start);
    return values;
  }

  const double step = (end - start) / static_cast<double>(count - 1);
  for (int i = 0; i < count; ++i) {
    values.push_back(start + step * static_cast<double>(i));
  }
  return values;
}

}  // namespace

int main()
{
  const auto t_f = linspace(0.0, M_PI_2, 20);
  const auto t_b = linspace(M_PI_2, 0.0, 20);
  const double r_f = 5.0;
  const double r_b = 2.0;

  std::vector<double> x_ref;
  std::vector<double> y_ref;
  std::vector<double> theta_ref;
  std::vector<double> gears;
  x_ref.reserve(39);
  y_ref.reserve(39);
  theta_ref.reserve(39);
  gears.reserve(38);

  for (std::size_t i = 0; i < t_f.size(); ++i) {
    x_ref.push_back(r_f * std::sin(t_f[i]));
    y_ref.push_back(r_f * (1.0 - std::cos(t_f[i])));
    theta_ref.push_back(t_f[i]);
    if (i + 1 < t_f.size()) {
      gears.push_back(1.0);
    }
  }

  for (std::size_t i = 1; i < t_b.size(); ++i) {
    x_ref.push_back(5.0 + r_b * (std::sin(t_b[i]) - 1.0));
    y_ref.push_back(3.0 + r_b * (1.0 - std::cos(t_b[i])));
    theta_ref.push_back(t_b[i]);
    gears.push_back(-1.0);
  }

  nonline_smoother::SmootherParams params;
  params.w_ref = 100.0;
  params.w_kappa = 0.1;
  params.w_dkappa = 10.0;
  params.w_ds = 1.0;
  params.max_kappa = 0.6;
  params.dynamic_weight = 100.0;
  params.max_num_iterations = 80;

  const nonline_smoother::CeresPathSmoother smoother(params);
  const auto result = smoother.solve(x_ref, y_ref, theta_ref, gears);

  std::cout << result.brief_report << std::endl;
  std::cout << "solve_time_ms=" << result.solve_time_ms << std::endl;
  std::cout << "final_cost=" << result.final_cost << std::endl;

  if (!result.success) {
    std::cerr << "smoother failed" << std::endl;
    return EXIT_FAILURE;
  }

  int cusp_count = 0;
  for (std::size_t i = 0; i < result.ds.size(); ++i) {
    if (std::abs(result.ds[i]) < 1e-6) {
      ++cusp_count;
      std::cout << "cusp_segment=" << i
                << " pose=(" << result.x[i] << ", " << result.y[i] << ")"
                << " kappa_jump=" << (result.kappa[i + 1] - result.kappa[i])
                << std::endl;
    }
  }

  if (cusp_count == 0) {
    std::cerr << "expected at least one cusp segment" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}