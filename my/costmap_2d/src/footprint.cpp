/*
 * Copyright (c) 2013, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "costmap_2d/footprint.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "costmap_2d/costmap_math.hpp"

namespace costmap_2d
{

std::pair<double, double> calculateMinAndMaxDistances(
  const std::vector<Point> & footprint)
{
  double min_dist = std::numeric_limits<double>::max();
  double max_dist = 0.0;

  if (footprint.size() <= 2) {
    return std::pair<double, double>(min_dist, max_dist);
  }

  for (unsigned int i = 0; i < footprint.size() - 1; ++i) {
    double vertex_dist = distance(0.0, 0.0, footprint[i].x, footprint[i].y);
    double edge_dist = distanceToLine(
      0.0, 0.0, footprint[i].x, footprint[i].y,
      footprint[i + 1].x, footprint[i + 1].y);
    min_dist = std::min(min_dist, std::min(vertex_dist, edge_dist));
    max_dist = std::max(max_dist, std::max(vertex_dist, edge_dist));
  }

  double vertex_dist = distance(0.0, 0.0, footprint.back().x, footprint.back().y);
  double edge_dist = distanceToLine(
    0.0, 0.0, footprint.back().x, footprint.back().y,
    footprint.front().x, footprint.front().y);
  min_dist = std::min(min_dist, std::min(vertex_dist, edge_dist));
  max_dist = std::max(max_dist, std::max(vertex_dist, edge_dist));

  return std::pair<double, double>(min_dist, max_dist);
}

void transformFootprint(
  double x, double y, double theta,
  const std::vector<Point> & footprint_spec,
  std::vector<Point> & oriented_footprint)
{
  oriented_footprint.resize(footprint_spec.size());
  double cos_th = cos(theta);
  double sin_th = sin(theta);
  for (unsigned int i = 0; i < footprint_spec.size(); ++i) {
    double new_x = x + (footprint_spec[i].x * cos_th - footprint_spec[i].y * sin_th);
    double new_y = y + (footprint_spec[i].x * sin_th + footprint_spec[i].y * cos_th);
    Point & new_pt = oriented_footprint[i];
    new_pt.x = new_x;
    new_pt.y = new_y;
  }
}

void padFootprint(std::vector<Point> & footprint, double padding)
{
  for (unsigned int i = 0; i < footprint.size(); i++) {
    Point & pt = footprint[i];
    pt.x += sign0(pt.x) * padding;
    pt.y += sign0(pt.y) * padding;
  }
}

std::vector<Point> makeFootprintFromRadius(double radius)
{
  std::vector<Point> points;

  int N = 16;
  Point pt;
  for (int i = 0; i < N; ++i) {
    double angle = i * 2 * M_PI / N;
    pt.x = cos(angle) * radius;
    pt.y = sin(angle) * radius;

    points.push_back(pt);
  }

  return points;
}

bool makeFootprintFromString(
  const std::string & footprint_string,
  std::vector<Point> & footprint)
{
  // Parse format: [[x1, y1], [x2, y2], ...]
  if (footprint_string.empty()) {
    return false;
  }

  std::vector<std::vector<float>> vvf;
  std::vector<float> current;
  int bracket_depth = 0;
  bool in_number = false;
  size_t num_start = 0;

  for (size_t i = 0; i < footprint_string.size(); ++i) {
    char c = footprint_string[i];
    if (c == '[') {
      bracket_depth++;
      if (bracket_depth == 2) {
        current.clear();
      }
    } else if (c == ']') {
      if (in_number) {
        float val = std::stof(footprint_string.substr(num_start, i - num_start));
        current.push_back(val);
        in_number = false;
      }
      if (bracket_depth == 2) {
        vvf.push_back(current);
      }
      bracket_depth--;
    } else if (bracket_depth == 2) {
      if (c >= '0' && c <= '9' || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
        if (!in_number) {
          num_start = i;
          in_number = true;
        }
      } else if (c == ',' || c == ' ') {
        if (in_number) {
          float val = std::stof(footprint_string.substr(num_start, i - num_start));
          current.push_back(val);
          in_number = false;
        }
      }
    }
  }

  if (in_number) {
    float val = std::stof(footprint_string.substr(num_start));
    current.push_back(val);
  }

  if (vvf.size() < 3) {
    return false;
  }

  footprint.reserve(vvf.size());
  for (unsigned int i = 0; i < vvf.size(); i++) {
    if (vvf[i].size() == 2) {
      Point point;
      point.x = vvf[i][0];
      point.y = vvf[i][1];
      point.z = 0;
      footprint.push_back(point);
    } else {
      return false;
    }
  }

  return true;
}

}  // namespace costmap_2d
