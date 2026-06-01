# C++ Ceres Nonlinear Smoother

This directory contains a standalone C++ implementation of the nonlinear path smoother, kept separate from the Python prototype in `my/nonline_smoother`.

## Key points

- The C++ solver is implemented in `ceres_nonlinear_smoother.cpp`.
- Each residual block only touches a local subset of variables:
  - reference tracking: `pose_i`
  - curvature regularization: `kappa_i`
  - transition dynamics: `pose_i`, `kappa_i`, `pose_{i+1}`, `kappa_{i+1}`, `control_i`
  - virtual cusp transition: `pose_i`, `pose_{i+1}`, `control_i`
- This preserves the banded sparse structure of the path optimization problem and allows Ceres to use `SPARSE_NORMAL_CHOLESKY` efficiently.

## Build

```bash
cd my/nonline_smoother_ceres
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --target nonline_smoother_cusp_demo -j4
```

The local `cmake/eigen3_compat` package is used to work around the Homebrew macOS mismatch where Ceres expects Eigen `3.4.0` while Homebrew currently installs Eigen `5.x` headers.

## Run the demo

```bash
cd my/nonline_smoother_ceres/build
ctest --output-on-failure -R nonline_smoother_cusp_demo
./nonline_smoother_cusp_demo
```

The demo recreates the existing cusp example from the Python prototype and checks that the optimized result retains at least one zero-length virtual segment.