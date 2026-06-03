# Kinematic Smoother

A standalone, ROS-independent library for kinematic path smoothing, extracted and simplified from `constrained_smoother`.

## Features
- A C++ kinematic path smoother built on Ceres and Eigen.
- A lightweight C++ A* planner plus ESDF utilities.
- Python bindings (pybind11).
- A Flask-based Web Lab for inspecting costmaps, planner output, and smoother behavior.

## Build Instructions
```bash
mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON -DBUILD_TESTS=ON
make -j
```

## Running Web App
```bash
./run_web_app.sh
```
