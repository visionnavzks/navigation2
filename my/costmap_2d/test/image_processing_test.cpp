// Copyright (c) 2023 Andrey Ryzhikov
// Ported to ROS-free costmap_2d library

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "costmap_2d/denoise/image_processing.hpp"
#include "costmap_2d/denoise/image.hpp"

using namespace costmap_2d;
using namespace imgproc_impl;

// Helper functions
template<class T>
Image<T> makeImage(size_t rows, size_t columns, std::vector<T> & buffer, size_t step = 0)
{
  step = std::max(step, columns);
  buffer.resize(rows * step);
  return Image<T>(rows, columns, buffer.data(), step);
}

template<class T>
Image<T> imageFromString(
  const std::string & s, std::vector<T> & buffer,
  const std::map<char, T> & codes = {{'.', 0}, {'x', 255}})
{
  const size_t side_size = static_cast<size_t>(std::sqrt(s.size()));
  if (size_t(side_size) * side_size != s.size()) {
    throw std::logic_error("Test data error: parseBinaryMatrix: Unexpected input string size");
  }
  const size_t step = static_cast<size_t>(side_size * 3);
  Image<T> image = makeImage(side_size, side_size, buffer, step);
  auto iter = s.begin();
  image.forEach(
    [&](T & pixel) {
      pixel = codes.at(*iter);
      ++iter;
    });
  return image;
}

inline bool isEqual(const Image<uint8_t> & a, const Image<uint8_t> & b)
{
  bool equal = a.rows() == b.rows() && a.columns() == b.columns();
  for (size_t row = 0; row < a.rows() && equal; ++row) {
    for (size_t column = 0; column < a.columns() && equal; ++column) {
      equal = a.row(row)[column] == b.row(row)[column];
    }
  }
  return equal;
}

std::map<char, uint8_t> makeLabelsMap(char max_symbol)
{
  std::map<char, uint8_t> labels_map = {{'.', 0}};
  for (char s = 'a'; s <= max_symbol; ++s) {
    labels_map.emplace(s, uint8_t(s - 'a' + 1));
  }
  return labels_map;
}

struct ImageProcTester : public ::testing::Test
{
protected:
  std::vector<uint8_t> image_buffer_bytes_;
  std::vector<uint8_t> image_buffer_bytes2_;
  std::vector<uint8_t> image_buffer_bytes3_;
  std::vector<uint16_t> image_buffer_words_;
};

// ===================== OutOfBounds Tests =====================

TEST(OutOfBounds, outOfBoundsAccess) {
  {
    out_of_bounds_policy::ReplaceToZero<uint8_t> c(nullptr, nullptr, 2);
    uint8_t * any_non_null = reinterpret_cast<uint8_t *>(&c);
    ASSERT_EQ(c.up(any_non_null), uint8_t(0));
  }
  {
    std::array<uint8_t, 3> data = {1, 2, 3};
    out_of_bounds_policy::ReplaceToZero<uint8_t> c(data.data(), data.data(), 2);
    auto left_out_of_bounds = std::prev(data.data());
    auto right_out_of_bounds = data.data() + data.size();

    ASSERT_EQ(c.up(left_out_of_bounds), uint8_t(0));
    ASSERT_EQ(c.up(right_out_of_bounds), uint8_t(0));
    ASSERT_EQ(c.down(left_out_of_bounds), uint8_t(0));
    ASSERT_EQ(c.down(right_out_of_bounds), uint8_t(0));
  }
}

// ===================== Histogram Tests =====================

TEST_F(ImageProcTester, calculateHistogramWithoutTruncation) {
  image_buffer_words_ = {0, 2, 1, 0, 3, 4, 1, 2, 0};
  Image<uint16_t> image = makeImage(3, 3, image_buffer_words_);
  const uint16_t max_bin_size = 3;
  const uint16_t max_value = 4;
  const auto hist = histogram(image, max_value, max_bin_size);

  const std::array<uint8_t, 5> expected = {3, 2, 2, 1, 1};
  ASSERT_EQ(hist.size(), expected.size());
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), hist.begin()));
}

TEST_F(ImageProcTester, calculateHistogramWithTruncation) {
  image_buffer_words_ = {0, 2, 1, 0, 3, 4, 1, 2, 0};
  Image<uint16_t> image = makeImage(3, 3, image_buffer_words_);
  const uint16_t max_bin_size = 2;
  const uint16_t max_value = 4;
  const auto hist = histogram(image, max_value, max_bin_size);

  const std::array<uint8_t, 5> expected = {2, 2, 2, 1, 1};
  ASSERT_EQ(hist.size(), expected.size());
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), hist.begin()));
}

TEST_F(ImageProcTester, calculateHistogramOfEmpty) {
  const uint16_t max_bin_size = 1;
  const uint16_t max_value = 0;
  Image<uint16_t> empty;

  const auto hist = histogram(empty, max_value, max_bin_size);
  ASSERT_TRUE(hist.empty());
}

// ===================== EquivalenceLabelTrees Tests =====================

TEST(EquivalenceLabelTrees, newLabelsTest) {
  EquivalenceLabelTrees<uint8_t> eq;
  eq.reset(10, 10, ConnectivityType::Way4);
  ASSERT_EQ(eq.makeLabel(), 1);
  ASSERT_EQ(eq.makeLabel(), 2);
  ASSERT_EQ(eq.makeLabel(), 3);
}

TEST(EquivalenceLabelTrees, unionTest) {
  EquivalenceLabelTrees<uint8_t> eq;
  eq.reset(10, 10, ConnectivityType::Way4);

  for (size_t i = 1; i < 6; ++i) {
    eq.makeLabel();
  }
  ASSERT_EQ(eq.unionTrees(4, 3), 3);
  ASSERT_EQ(eq.unionTrees(5, 3), 3);
  ASSERT_EQ(eq.unionTrees(4, 1), 1);
  ASSERT_EQ(eq.unionTrees(2, 5), 1);
}

// ===================== Connected Components Tests =====================

struct ConnectedComponentsTester : public ImageProcTester
{
protected:
  MemoryBuffer buffer_;
  imgproc_impl::EquivalenceLabelTrees<uint8_t> label_trees_;
  static const uint8_t BACKGROUND_CODE = 0;
  static const uint8_t FOREGROUND_CODE = 255;

  inline static bool isBackground(uint8_t pixel) {
    return pixel == BACKGROUND_CODE;
  }
};

TEST_F(ConnectedComponentsTester, way4EmptyTest) {
  Image<uint8_t> empty;
  uint8_t total_labels;
  connectedComponents<ConnectivityType::Way4>(
    empty, buffer_, label_trees_,
    isBackground, total_labels);
  ASSERT_EQ(total_labels, uint8_t(0));
}

TEST_F(ConnectedComponentsTester, way4SinglePixelTest) {
  Image<uint8_t> input = makeImage(1, 1, image_buffer_bytes_);
  uint8_t total_labels;
  {
    input.row(0)[0] = BACKGROUND_CODE;
    const auto result = connectedComponents<ConnectivityType::Way4>(
      input, buffer_, label_trees_,
      isBackground, total_labels);
    ASSERT_EQ(result.row(0)[0], 0);
    ASSERT_EQ(total_labels, 1);
  }
  {
    input.row(0)[0] = FOREGROUND_CODE;
    const auto result = connectedComponents<ConnectivityType::Way4>(
      input, buffer_, label_trees_,
      isBackground, total_labels);
    ASSERT_EQ(result.row(0)[0], 1);
    ASSERT_EQ(total_labels, 2);
  }
}

TEST_F(ConnectedComponentsTester, way4ImageSmallTest) {
  {
    Image<uint8_t> input = makeImage(1, 2, image_buffer_bytes_);
    uint8_t total_labels;
    input.row(0)[0] = BACKGROUND_CODE;
    input.row(0)[1] = FOREGROUND_CODE;
    const auto result = connectedComponents<ConnectivityType::Way4>(
      input, buffer_, label_trees_,
      isBackground, total_labels);
    ASSERT_EQ(total_labels, uint8_t(2));
    ASSERT_EQ(result.row(0)[0], 0);
    ASSERT_EQ(result.row(0)[1], 1);
  }
  {
    Image<uint8_t> input = makeImage(2, 1, image_buffer_bytes_);
    uint8_t total_labels;
    input.row(0)[0] = BACKGROUND_CODE;
    input.row(1)[0] = FOREGROUND_CODE;
    const auto result = connectedComponents<ConnectivityType::Way4>(
      input, buffer_, label_trees_,
      isBackground, total_labels);
    ASSERT_EQ(total_labels, uint8_t(2));
    ASSERT_EQ(result.row(0)[0], 0);
    ASSERT_EQ(result.row(1)[0], 1);
  }
}

TEST_F(ConnectedComponentsTester, way4ImageStepsTest) {
  const Image<uint8_t> input = imageFromString<uint8_t>(
    "..xx"
    ".xx."
    "xx.."
    "....", image_buffer_bytes_);
  const Image<uint8_t> expected_labels = imageFromString<uint8_t>(
    "..xx"
    ".xx."
    "xx.."
    "....", image_buffer_bytes2_);
  uint8_t total_labels;
  const auto result = connectedComponents<ConnectivityType::Way4>(
    input, buffer_, label_trees_,
    isBackground, total_labels);
  ASSERT_EQ(total_labels, uint8_t(2));
}

TEST_F(ConnectedComponentsTester, way8ImageStepsTest) {
  const Image<uint8_t> input = imageFromString<uint8_t>(
    "....xx"
    "..xx.."
    "xx...."
    "...xx."
    ".....x"
    "....x.", image_buffer_bytes_);
  uint8_t total_labels;
  const auto result = connectedComponents<ConnectivityType::Way8>(
    input, buffer_, label_trees_,
    isBackground, total_labels);
  ASSERT_EQ(total_labels, uint8_t(3));
}

// ===================== Morphology Tests =====================

ShapeBuffer3x3 shape_buffer{};
const Image<uint8_t> cross_shape = createShape(shape_buffer, ConnectivityType::Way4);

uint8_t max_list(std::initializer_list<uint8_t> lst) {
  return std::max(lst);
}

TEST_F(ImageProcTester, emptyImage) {
  Image<uint8_t> input;
  Image<uint8_t> output;
  ASSERT_NO_THROW(morphologyOperation(input, output, cross_shape, max_list));
}

TEST_F(ImageProcTester, wrongShapeSize) {
  Image<uint8_t> input = makeImage(1, 1, image_buffer_bytes_);
  Image<uint8_t> output = makeImage(1, 1, image_buffer_bytes2_);
  ASSERT_THROW(
    morphologyOperation(input, output, makeImage(2, 2, image_buffer_bytes3_), max_list),
    std::logic_error);
}

TEST_F(ImageProcTester, singlePixelImage) {
  image_buffer_bytes_ = {255};
  Image<uint8_t> input = makeImage(1, 1, image_buffer_bytes_);
  Image<uint8_t> output = makeImage(1, 1, image_buffer_bytes2_);
  morphologyOperation(input, output, cross_shape, max_list);
  ASSERT_EQ(output.row(0)[0], 0);
}

TEST_F(ImageProcTester, cornersImage) {
  const Image<uint8_t> input = imageFromString<uint8_t>(
    "x..x"
    "...."
    "...."
    "x..x", image_buffer_bytes_);
  Image<uint8_t> expected = imageFromString<uint8_t>(
    ".xx."
    "x..x"
    "x..x"
    ".xx.", image_buffer_bytes2_);
  Image<uint8_t> output = makeImage(input.rows(), input.columns(), image_buffer_bytes3_);
  morphologyOperation(input, output, cross_shape, max_list);
  ASSERT_TRUE(isEqual(output, expected));
}

TEST_F(ImageProcTester, horizontalBordersImage) {
  const Image<uint8_t> input = imageFromString<uint8_t>(
    "x..x"
    "x..x"
    "x..x"
    "x..x", image_buffer_bytes_);
  Image<uint8_t> expected = imageFromString<uint8_t>(
    "xxxx"
    "xxxx"
    "xxxx"
    "xxxx", image_buffer_bytes2_);
  Image<uint8_t> output = makeImage(input.rows(), input.columns(), image_buffer_bytes3_);
  morphologyOperation(input, output, cross_shape, max_list);
  ASSERT_TRUE(isEqual(output, expected));
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
