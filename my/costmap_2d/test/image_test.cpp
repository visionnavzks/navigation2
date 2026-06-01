// Copyright (c) 2023 Andrey Ryzhikov
// Ported to ROS-free costmap_2d library

#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>

#include "costmap_2d/denoise/image.hpp"

using namespace costmap_2d;

// Helper functions (from original image_tests_helper.hpp)
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
      try {
        pixel = codes.at(*iter);
        ++iter;
      } catch (...) {
        throw std::logic_error(
          "Test data error: parseBinaryMatrix: Unexpected symbol: " +
          std::string(1, *iter));
      }
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

struct ImageTester : public ::testing::Test
{
protected:
  std::vector<uint8_t> image_buffer_bytes;
  std::vector<uint16_t> image_buffer_words;
};

TEST_F(ImageTester, emptyProps) {
  Image<uint8_t> empty;
  ASSERT_EQ(empty.rows(), 0ul);
  ASSERT_EQ(empty.columns(), 0ul);
  ASSERT_EQ(empty.step(), 0ul);
}

TEST_F(ImageTester, memoryAccess) {
  std::array<uint8_t, 7> buffer{};
  for (uint8_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = i;
  }
  Image<uint8_t> wrapper(2, 3, buffer.data(), 4);

  ASSERT_EQ(wrapper.row(0), buffer.data());
  ASSERT_EQ(wrapper.row(1), buffer.data() + 4);
}

TEST_F(ImageTester, forEach) {
  Image<uint8_t> image = makeImage(3, 2, image_buffer_bytes);
  const uint8_t non_zero_initial_value = 42;
  uint8_t current(non_zero_initial_value);

  image.forEach(
    [&](uint8_t & pixel) {
      pixel = current++;
    });

  for (size_t i = 0; i < image.rows(); ++i) {
    for (size_t j = 0; j < image.columns(); ++j) {
      ASSERT_EQ(image.row(i)[j], non_zero_initial_value + i * image.columns() + j);
    }
  }
}

TEST_F(ImageTester, convert) {
  image_buffer_words = {1, 2, 3, 4, 5, 6};
  Image<uint16_t> source = makeImage(2, 3, image_buffer_words);
  Image<uint8_t> target = makeImage(2, 3, image_buffer_bytes);

  source.convert(
    target, [](uint16_t s, uint8_t & t) {
      t = s * 2;
    });

  const std::array<uint8_t, 6> expected = {2, 4, 6, 8, 10, 12};
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), image_buffer_bytes.begin()));
}

TEST_F(ImageTester, convertDifferentSizes) {
  Image<uint16_t> source = makeImage(2, 3, image_buffer_words);
  Image<uint8_t> target = makeImage(3, 2, image_buffer_bytes);
  auto do_nothing = [](uint16_t /*src*/, uint8_t & /*trg*/) {};

  ASSERT_THROW((source.convert(target, do_nothing)), std::logic_error);
}

TEST_F(ImageTester, convertEmptyImages) {
  const Image<uint16_t> source;
  Image<uint8_t> target;
  auto shouldn_t_be_called = [](uint16_t /*src*/, uint8_t & /*trg*/) {
      throw std::logic_error("");
    };

  ASSERT_NO_THROW((source.convert(target, shouldn_t_be_called)));
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
