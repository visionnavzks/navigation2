// Copyright (c) 2024
// Licensed under the Apache License, Version 2.0
//
// Standalone executable for the kinematic smoother.
// Reads a JSON problem from stdin, solves it, writes JSON result to stdout.
//
// Usage:
//   echo '{ ... }' | ./kinematic_smoother_node
//
// Input JSON fields:
//   x_ref, y_ref, theta_ref : arrays of doubles (reference path)
//   gears                   : array of doubles (+1 / -1), length = len(x_ref)-1
//   params                  : object with algorithm parameters
//   obstacles (optional)    : array of {x_min, y_min, x_max, y_max}
//   esdf (optional)         : {resolution, origin_x, origin_y, width, height, data:[...]}

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kinematic_smoother/kinematic_smoother.hpp"

// ============================================================================
// Minimal JSON parser (supports the subset we need)
// ============================================================================
namespace json
{

enum Type { Null, Bool, Number, String, Array, Object };

struct Value
{
  Type type = Null;
  double number = 0.0;
  bool boolean = false;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  const Value & operator[](const std::string & key) const
  {
    for (auto & [k, v] : obj) {
      if (k == key) {return v;}
    }
    static Value null_val;
    return null_val;
  }

  const Value & operator[](size_t idx) const { return arr.at(idx); }

  double asNumber(double def = 0.0) const
  {
    return type == Number ? number : def;
  }

  bool asBool(bool def = false) const
  {
    return type == Bool ? boolean : def;
  }

  bool has(const std::string & key) const
  {
    for (auto & [k, v] : obj) {
      if (k == key) {return true;}
    }
    return false;
  }

  std::vector<double> asNumberArray() const
  {
    std::vector<double> out;
    for (auto & v : arr) { out.push_back(v.asNumber()); }
    return out;
  }
};

class Parser
{
public:
  explicit Parser(const std::string & s) : s_(s), pos_(0) {}

  Value parse()
  {
    skipWS();
    return parseValue();
  }

private:
  const std::string & s_;
  size_t pos_;

  char peek() { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  char next() { return pos_ < s_.size() ? s_[pos_++] : '\0'; }

  void skipWS()
  {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
           s_[pos_] == '\n' || s_[pos_] == '\r'))
    {
      ++pos_;
    }
  }

  Value parseValue()
  {
    skipWS();
    char c = peek();
    if (c == '"') {return parseString();}
    if (c == '{') {return parseObject();}
    if (c == '[') {return parseArray();}
    if (c == 't' || c == 'f') {return parseBool();}
    if (c == 'n') { pos_ += 4; return Value(); }
    return parseNumber();
  }

  Value parseString()
  {
    Value v;
    v.type = String;
    next(); // skip "
    while (peek() != '"') {
      if (peek() == '\\') { next(); }
      v.str += next();
    }
    next(); // skip "
    return v;
  }

  Value parseNumber()
  {
    Value v;
    v.type = Number;
    size_t start = pos_;
    if (peek() == '-') {next();}
    // Consume digits, decimal point
    while (std::isdigit(peek())) {next();}
    if (peek() == '.') {
      next();
      while (std::isdigit(peek())) {next();}
    }
    // Consume exponent
    if (peek() == 'e' || peek() == 'E') {
      next();
      if (peek() == '+' || peek() == '-') {next();}
      while (std::isdigit(peek())) {next();}
    }
    v.number = std::stod(s_.substr(start, pos_ - start));
    return v;
  }

  Value parseBool()
  {
    Value v;
    v.type = Bool;
    if (peek() == 't') { v.boolean = true; pos_ += 4; }
    else { v.boolean = false; pos_ += 5; }
    return v;
  }

  Value parseArray()
  {
    Value v;
    v.type = Array;
    next(); // [
    skipWS();
    if (peek() == ']') { next(); return v; }
    while (true) {
      v.arr.push_back(parseValue());
      skipWS();
      if (peek() == ',') { next(); skipWS(); continue; }
      break;
    }
    next(); // ]
    return v;
  }

  Value parseObject()
  {
    Value v;
    v.type = Object;
    next(); // {
    skipWS();
    if (peek() == '}') { next(); return v; }
    while (true) {
      skipWS();
      auto key = parseString();
      skipWS();
      next(); // :
      auto val = parseValue();
      v.obj.emplace_back(key.str, std::move(val));
      skipWS();
      if (peek() == ',') { next(); continue; }
      break;
    }
    next(); // }
    return v;
  }
};

Value parse(const std::string & s)
{
  Parser p(s);
  return p.parse();
}

}  // namespace json

// ============================================================================
// JSON writer helpers
// ============================================================================

static void writeArray(std::ostream & os, const std::vector<double> & v)
{
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {os << ",";}
    os << v[i];
  }
  os << "]";
}

// ============================================================================
// main
// ============================================================================
int main()
{
  // Read all of stdin
  std::ostringstream ss;
  ss << std::cin.rdbuf();
  std::string input = ss.str();

  auto root = json::parse(input);

  // --- Extract reference path ---
  auto x_ref     = root["x_ref"].asNumberArray();
  auto y_ref     = root["y_ref"].asNumberArray();
  auto theta_ref = root["theta_ref"].asNumberArray();
  auto gears_arr = root["gears"].asNumberArray();

  if (x_ref.empty() || y_ref.empty() || theta_ref.empty()) {
    std::cout << R"({"success":false,"message":"Empty reference path"})";
    return 0;
  }

  // --- Extract parameters ---
  kinematic_smoother::SmootherParams params;
  auto & p = root["params"];
  if (p.type == json::Object) {
    params.max_kappa      = p["max_kappa"].asNumber(params.max_kappa);
    params.w_ref          = p["w_ref"].asNumber(params.w_ref);
    params.w_dkappa       = p["w_dkappa"].asNumber(params.w_dkappa);
    params.w_kappa        = p["w_kappa"].asNumber(params.w_kappa);
    params.w_ds           = p["w_ds"].asNumber(params.w_ds);
    params.w_kinematic    = p["w_kinematic"].asNumber(params.w_kinematic);
    params.target_ds      = p["target_ds"].asNumber(params.target_ds);
    params.ds_min_ratio   = p["ds_min_ratio"].asNumber(params.ds_min_ratio);
    params.ds_max_ratio   = p["ds_max_ratio"].asNumber(params.ds_max_ratio);
    params.max_iterations = static_cast<int>(p["max_iterations"].asNumber(params.max_iterations));
    params.tolerance      = p["tolerance"].asNumber(params.tolerance);
    params.debug          = p["debug"].asBool(params.debug);

    params.fix_start_kappa = p["fix_start_kappa"].asBool(params.fix_start_kappa);
    params.kappa_start     = p["kappa_start"].asNumber(params.kappa_start);

    params.w_esdf              = p["w_esdf"].asNumber(params.w_esdf);
    params.esdf_safe_distance  = p["esdf_safe_distance"].asNumber(params.esdf_safe_distance);
  }

  // --- Build ESDF (optional) ---
  kinematic_smoother::ESDF esdf;
  if (root.has("obstacles") && root["obstacles"].type == json::Array) {
    // Build ESDF from rectangle obstacles
    std::vector<kinematic_smoother::RectObstacle> obstacles;
    for (auto & obs : root["obstacles"].arr) {
      obstacles.push_back({
        obs["x_min"].asNumber(), obs["y_min"].asNumber(),
        obs["x_max"].asNumber(), obs["y_max"].asNumber()
      });
    }

    // Determine grid bounds from path + margin
    double xmin = *std::min_element(x_ref.begin(), x_ref.end()) - 5.0;
    double xmax = *std::max_element(x_ref.begin(), x_ref.end()) + 5.0;
    double ymin = *std::min_element(y_ref.begin(), y_ref.end()) - 5.0;
    double ymax = *std::max_element(y_ref.begin(), y_ref.end()) + 5.0;
    double res = 0.1;
    int w = static_cast<int>((xmax - xmin) / res) + 1;
    int h = static_cast<int>((ymax - ymin) / res) + 1;
    esdf.build(res, xmin, ymin, w, h, obstacles);
  } else if (root.has("esdf") && root["esdf"].type == json::Object) {
    auto & e = root["esdf"];
    esdf.buildFromData(
      e["resolution"].asNumber(0.1),
      e["origin_x"].asNumber(0), e["origin_y"].asNumber(0),
      static_cast<int>(e["width"].asNumber(0)),
      static_cast<int>(e["height"].asNumber(0)),
      e["data"].asNumberArray());
  }

  // --- Solve ---
  kinematic_smoother::KinematicSmoother smoother;
  auto res = smoother.solve(x_ref, y_ref, theta_ref, gears_arr, params,
                            esdf.valid() ? &esdf : nullptr);

  // --- Write JSON output ---
  std::cout << "{";
  std::cout << "\"success\":" << (res.success ? "true" : "false") << ",";
  std::cout << "\"solve_time_ms\":" << res.solve_time_ms << ",";
  std::cout << "\"target_ds_mag\":" << res.target_ds_mag << ",";

  std::cout << "\"x_opt\":"; writeArray(std::cout, res.x_opt); std::cout << ",";
  std::cout << "\"y_opt\":"; writeArray(std::cout, res.y_opt); std::cout << ",";
  std::cout << "\"theta_opt\":"; writeArray(std::cout, res.theta_opt); std::cout << ",";
  std::cout << "\"kappa_opt\":"; writeArray(std::cout, res.kappa_opt); std::cout << ",";
  std::cout << "\"ds_opt\":"; writeArray(std::cout, res.ds_opt); std::cout << ",";
  std::cout << "\"dkappa_opt\":"; writeArray(std::cout, res.dkappa_opt); std::cout << ",";
  std::cout << "\"gears_opt\":"; writeArray(std::cout, res.gears_opt); std::cout << ",";

  std::cout << "\"costs\":{";
  std::cout << "\"total\":" << res.costs.total << ",";
  std::cout << "\"ref\":" << res.costs.ref << ",";
  std::cout << "\"smooth\":" << res.costs.smooth << ",";
  std::cout << "\"kappa\":" << res.costs.kappa << ",";
  std::cout << "\"ds\":" << res.costs.ds << ",";
  std::cout << "\"kinematic\":" << res.costs.kinematic << ",";
  std::cout << "\"esdf\":" << res.costs.esdf;
  std::cout << "}";

  std::cout << "}" << std::endl;

  return 0;
}
