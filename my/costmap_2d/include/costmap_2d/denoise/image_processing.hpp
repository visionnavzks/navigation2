// Copyright (c) 2023 Andrey Ryzhikov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COSTMAP_2D__DENOISE__IMAGE_PROCESSING_HPP_
#define COSTMAP_2D__DENOISE__IMAGE_PROCESSING_HPP_

#include "image.hpp"
#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <string>
#include <utility>

namespace costmap_2d
{

enum class ConnectivityType : int
{
  Way4 = 4,
  Way8 = 8
};

class MemoryBuffer
{
public:
  inline ~MemoryBuffer() {reset();}

  template<class T>
  T * get(std::size_t count);

private:
  inline void reset();
  inline void allocate(size_t bytes);

private:
  void * data_{};
  size_t size_{};
};

namespace imgproc_impl
{
template<class Label>
class EquivalenceLabelTrees;

template<class AggregateFn>
void morphologyOperation(
  const Image<uint8_t> & input, Image<uint8_t> & output,
  const Image<uint8_t> & shape, AggregateFn aggregate);

using ShapeBuffer3x3 = std::array<uint8_t, 9>;
inline Image<uint8_t> createShape(ShapeBuffer3x3 & buffer, ConnectivityType connectivity);
}  // namespace imgproc_impl

template<class Max>
inline void dilate(
  const Image<uint8_t> & input, Image<uint8_t> & output,
  ConnectivityType connectivity, Max && max_function)
{
  using namespace imgproc_impl;
  ShapeBuffer3x3 shape_buffer;
  Image<uint8_t> shape = createShape(shape_buffer, connectivity);
  morphologyOperation(input, output, shape, max_function);
}

template<ConnectivityType connectivity, class Label, class IsBg>
std::pair<Image<Label>, Label> connectedComponents(
  const Image<uint8_t> & image, MemoryBuffer & buffer,
  imgproc_impl::EquivalenceLabelTrees<Label> & label_trees,
  IsBg && is_background);

// Implementation

template<class T>
T * MemoryBuffer::get(std::size_t count)
{
  static_assert(
    alignof(std::max_align_t) >= alignof(T),
    "T alignment is more than the fundamental alignment of the platform");

  const size_t required_bytes = sizeof(T) * count;

  if (size_ < required_bytes) {
    allocate(required_bytes);
  }
  return static_cast<T *>(data_);
}

void MemoryBuffer::reset()
{
  ::operator delete(data_);
  size_ = 0;
}

void MemoryBuffer::allocate(size_t bytes)
{
  reset();
  data_ = ::operator new(bytes);
  size_ = bytes;
}

namespace imgproc_impl
{

template<class T, class Bin>
std::vector<Bin>
histogram(const Image<T> & image, T image_max, Bin bin_max)
{
  if (image.empty()) {
    return {};
  }
  std::vector<Bin> histogram(size_t(image_max) + 1);

  auto add_pixel_value = [&histogram, bin_max](T pixel) {
      auto & h = histogram[pixel];
      h = std::min(Bin(h + 1), bin_max);
    };

  image.forEach(add_pixel_value);
  return histogram;
}

namespace out_of_bounds_policy
{

template<class T>
struct DoNothing
{
  T & up(T * v) const {return *v;}
  T & down(T * v) const {return *v;}
};

template<class T>
class ReplaceToZero
{
public:
  ReplaceToZero(const T * up_row_start, const T * down_row_start, size_t columns)
  : up_row_start_{up_row_start}, up_row_end_{up_row_start + columns},
    down_row_start_{down_row_start}, down_row_end_{down_row_start + columns} {}

  T & up(T * v)
  {
    if (up_row_start_ == nullptr) {
      return zero_;
    }
    return replaceOutOfBounds(v, up_row_start_, up_row_end_);
  }

  T & down(T * v)
  {
    return replaceOutOfBounds(v, down_row_start_, down_row_end_);
  }

private:
  T & replaceOutOfBounds(T * v, const T * begin, const T * end)
  {
    if (v < begin || v >= end) {
      return zero_;
    }
    return *v;
  }

  const T * up_row_start_;
  const T * up_row_end_;
  const T * down_row_start_;
  const T * down_row_end_;
  T zero_{};
};

}  // namespace out_of_bounds_policy

template<class T, template<class> class Border>
class Window
{
public:
  inline Window(T * up_row, T * down_row, Border<T> border = {})
  : up_row_{up_row}, down_row_{down_row}, border_{border} {}

  inline T & a() {return border_.up(up_row_ - 1);}
  inline T & b() {return border_.up(up_row_);}
  inline T & c() {return border_.up(up_row_ + 1);}
  inline T & d() {return border_.down(down_row_ - 1);}
  inline T & e() {return *down_row_;}
  inline const T * anchor() const {return down_row_;}

  inline void next()
  {
    ++up_row_;
    ++down_row_;
  }

private:
  T * up_row_;
  T * down_row_;
  Border<T> border_;
};

template<class T>
T * dropConst(const T * ptr)
{
  return const_cast<T *>(ptr);
}

template<class T>
Window<T, out_of_bounds_policy::ReplaceToZero> makeSafeWindow(
  const T * up_row, const T * down_row, size_t columns, size_t offset = 0)
{
  return {
    dropConst(up_row) + offset, dropConst(down_row) + offset,
    out_of_bounds_policy::ReplaceToZero<T>{up_row, down_row, columns}
  };
}

template<class T>
Window<T, out_of_bounds_policy::DoNothing> makeUnsafeWindow(const T * up_row, const T * down_row)
{
  return {dropConst(up_row), dropConst(down_row)};
}

struct EquivalenceLabelTreesBase
{
  virtual ~EquivalenceLabelTreesBase() = default;
};

struct LabelOverflow : public std::runtime_error
{
  explicit LabelOverflow(const std::string & message)
  : std::runtime_error(message) {}
};

template<class Label>
class EquivalenceLabelTrees : public EquivalenceLabelTreesBase
{
public:
  void reset(const size_t rows, const size_t columns, ConnectivityType connectivity)
  {
    const size_t max_labels_count = maxLabels(rows, columns, connectivity);
    labels_size_ = static_cast<Label>(
      std::min(max_labels_count, size_t(std::numeric_limits<Label>::max()))
    );

    try {
      labels_.reserve(labels_size_);
    } catch (...) {
    }

    labels_.clear();
    labels_.resize(1, 0);
    next_free_ = 1;
  }

  Label makeLabel()
  {
    if (next_free_ == labels_size_) {
      throw LabelOverflow("EquivalenceLabelTrees: Can't create new label");
    }
    labels_.push_back(next_free_);
    return next_free_++;
  }

  Label unionTrees(Label i, Label j)
  {
    Label root = findRoot(i);

    if (i != j) {
      Label root_j = findRoot(j);
      root = std::min(root, root_j);
      setRoot(j, root);
    }
    setRoot(i, root);
    return root;
  }

  const std::vector<Label> & getLabels()
  {
    Label k = 1;
    for (Label i = 1; i < next_free_; ++i) {
      if (labels_[i] < i) {
        labels_[i] = labels_[labels_[i]];
      } else {
        labels_[i] = k;
        ++k;
      }
    }
    labels_.resize(k);
    return labels_;
  }

private:
  static size_t maxLabels(const size_t rows, const size_t columns, ConnectivityType connectivity)
  {
    size_t max_labels{};

    if (connectivity == ConnectivityType::Way4) {
      max_labels = (rows * columns) / 2 + 1;
    } else {
      max_labels = (rows * columns) / 3 + 1;
    }
    ++max_labels;
    max_labels = std::min(max_labels, size_t(std::numeric_limits<Label>::max()));
    return max_labels;
  }

  Label findRoot(Label i)
  {
    Label root = i;
    for (; labels_[root] < root; root = labels_[root]) {}
    return root;
  }

  void setRoot(Label i, Label root)
  {
    while (labels_[i] < i) {
      auto j = labels_[i];
      labels_[i] = root;
      i = j;
    }
    labels_[i] = root;
  }

private:
  std::vector<Label> labels_;
  Label labels_size_{};
  Label next_free_{};
};

template<ConnectivityType connectivity>
struct ProcessPixel;

template<>
struct ProcessPixel<ConnectivityType::Way8>
{
  template<class ImageWindow, class LabelsWindow, class Label, class IsBg>
  static void pass(
    ImageWindow & image, LabelsWindow & label, EquivalenceLabelTrees<Label> & eq_trees,
    IsBg && is_bg)
  {
    Label & current = label.e();

    if (!is_bg(image.e())) {
      if (label.b()) {
        current = label.b();
      } else {
        if (!is_bg(image.c())) {
          if (!is_bg(image.a())) {
            current = eq_trees.unionTrees(label.c(), label.a());
          } else {
            if (!is_bg(image.d())) {
              current = eq_trees.unionTrees(label.c(), label.d());
            } else {
              current = label.c();
            }
          }
        } else {
          if (!is_bg(image.a())) {
            current = label.a();
          } else {
            if (!is_bg(image.d())) {
              current = label.d();
            } else {
              current = eq_trees.makeLabel();
            }
          }
        }
      }
    } else {
      current = 0;
    }
  }
};

template<>
struct ProcessPixel<ConnectivityType::Way4>
{
  template<class ImageWindow, class LabelsWindow, class Label, class IsBg>
  static void pass(
    ImageWindow & image, LabelsWindow & label, EquivalenceLabelTrees<Label> & eq_trees,
    IsBg && is_bg)
  {
    Label & current = label.e();

    if (!is_bg(image.e())) {
      if (!is_bg(image.b())) {
        if (!is_bg(image.d())) {
          current = eq_trees.unionTrees(label.d(), label.b());
        } else {
          current = label.b();
        }
      } else {
        if (!is_bg(image.d())) {
          current = label.d();
        } else {
          current = eq_trees.makeLabel();
        }
      }
    } else {
      current = 0;
    }
  }
};

template<class Apply>
void probeRows(
  const Image<uint8_t> & input, size_t first_input_row,
  Image<uint8_t> & output, size_t first_output_row,
  const uint8_t * shape, Apply touch_fn)
{
  const size_t rows = input.rows() - std::max(first_input_row, first_output_row);
  const size_t columns = input.columns();

  auto apply_shape = [&shape](uint8_t value, uint8_t index) -> uint8_t {
      return value & shape[index];
    };

  auto get_input_row = [&input, first_input_row](size_t row) {
      return input.row(row + first_input_row);
    };
  auto get_output_row = [&output, first_output_row](size_t row) {
      return output.row(row + first_output_row);
    };

  if (columns == 1) {
    for (size_t i = 0; i < rows; ++i) {
      auto overlay = {uint8_t(0), apply_shape(*get_input_row(i), 1), uint8_t(0)};
      touch_fn(*get_output_row(i), overlay);
    }
  } else {
    for (size_t i = 0; i < rows; ++i) {
      const uint8_t * in = get_input_row(i);
      const uint8_t * last_column_pixel = in + columns - 1;
      uint8_t * out = get_output_row(i);

      {
        auto overlay = {uint8_t(0), apply_shape(*in, 1), apply_shape(*(in + 1), 2)};
        touch_fn(*out, overlay);
        ++in;
        ++out;
      }

      for (; in != last_column_pixel; ++in, ++out) {
        auto overlay = {
          apply_shape(*(in - 1), 0),
          apply_shape(*(in), 1),
          apply_shape(*(in + 1), 2)
        };
        touch_fn(*out, overlay);
      }

      {
        auto overlay = {apply_shape(*(in - 1), 0), apply_shape(*(in), 1), uint8_t(0)};
        touch_fn(*out, overlay);
        ++in;
        ++out;
      }
    }
  }
}

template<class AggregateFn>
void morphologyOperation(
  const Image<uint8_t> & input, Image<uint8_t> & output,
  const Image<uint8_t> & shape, AggregateFn aggregate)
{
  if (input.rows() != output.rows() || input.columns() != output.columns()) {
    throw std::logic_error(
            "morphologyOperation: the sizes of the input and output images are different");
  }

  if (shape.rows() != 3 || shape.columns() != 3) {
    throw std::logic_error("morphologyOperation: wrong shape size");
  }

  if (input.empty()) {
    return;
  }

  auto set = [&](uint8_t & res, std::initializer_list<uint8_t> lst) {res = aggregate(lst);};
  auto update = [&](uint8_t & res, std::initializer_list<uint8_t> lst) {
      res = aggregate({res, aggregate(lst), 0});
    };

  probeRows(input, 0, output, 0, shape.row(1), set);

  if (input.rows() > 1) {
    probeRows(input, 0, output, 1, shape.row(0), update);
    probeRows(input, 1, output, 0, shape.row(2), update);
  }
}

Image<uint8_t> createShape(ShapeBuffer3x3 & buffer, ConnectivityType connectivity)
{
  static constexpr uint8_t u = 255;
  static constexpr uint8_t i = 0;

  if (connectivity == ConnectivityType::Way8) {
    buffer = {
      u, u, u,
      u, i, u,
      u, u, u};
  } else {
    buffer = {
      i, u, i,
      u, i, u,
      i, u, i};
  }
  return Image<uint8_t>(3, 3, buffer.data(), 3);
}

template<ConnectivityType connectivity, class Label, class IsBg>
Label connectedComponentsImpl(
  const Image<uint8_t> & image, Image<Label> & labels,
  imgproc_impl::EquivalenceLabelTrees<Label> & label_trees, const IsBg & is_background)
{
  using namespace imgproc_impl;
  using PixelPass = ProcessPixel<connectivity>;

  {
    auto img = makeSafeWindow<uint8_t>(nullptr, image.row(0), image.columns());
    auto lbl = makeSafeWindow<Label>(nullptr, labels.row(0), image.columns());

    const uint8_t * first_row_end = image.row(0) + image.columns();

    for (; img.anchor() < first_row_end; img.next(), lbl.next()) {
      PixelPass::pass(img, lbl, label_trees, is_background);
    }
  }

  for (size_t row = 0; row < image.rows() - 1; ++row) {
    Window<Label, out_of_bounds_policy::DoNothing> label_mask{labels.row(row), labels.row(row + 1)};

    auto up = image.row(row);
    auto current = image.row(row + 1);

    {
      auto img = makeSafeWindow(up, current, image.columns());
      PixelPass::pass(img, label_mask, label_trees, is_background);
    }

    label_mask.next();

    auto img = makeUnsafeWindow(std::next(up), std::next(current));
    const uint8_t * current_row_last_element = current + image.columns() - 1;

    for (; img.anchor() < current_row_last_element; img.next(), label_mask.next()) {
      PixelPass::pass(img, label_mask, label_trees, is_background);
    }

    if (image.columns() > 1) {
      auto last_img = makeSafeWindow(up, current, image.columns(), image.columns() - 1);
      auto last_label = makeSafeWindow(
        labels.row(row), labels.row(row + 1),
        image.columns(), image.columns() - 1);
      PixelPass::pass(last_img, last_label, label_trees, is_background);
    }
  }

  const std::vector<Label> & labels_map = label_trees.getLabels();

  labels.forEach(
    [&](Label & l) {
      l = labels_map[l];
    });
  return labels_map.size();
}

class GroupsRemover
{
public:
  GroupsRemover()
  {
    label_trees_ = std::make_unique<imgproc_impl::EquivalenceLabelTrees<uint16_t>>();
  }

  template<class IsBg>
  void removeGroups(
    Image<uint8_t> & image, MemoryBuffer & buffer,
    ConnectivityType group_connectivity_type, size_t minimal_group_size,
    const IsBg & is_background) const
  {
    if (group_connectivity_type == ConnectivityType::Way4) {
      removeGroupsPickLabelType<ConnectivityType::Way4>(
        image, buffer, minimal_group_size,
        is_background);
    } else {
      removeGroupsPickLabelType<ConnectivityType::Way8>(
        image, buffer, minimal_group_size,
        is_background);
    }
  }

private:
  template<ConnectivityType connectivity, class IsBg>
  void removeGroupsPickLabelType(
    Image<uint8_t> & image, MemoryBuffer & buffer,
    size_t minimal_group_size, const IsBg & is_background) const
  {
    bool success{};
    auto label_trees16 =
      dynamic_cast<imgproc_impl::EquivalenceLabelTrees<uint16_t> *>(label_trees_.get());

    if (label_trees16) {
      success = tryRemoveGroupsWithLabelType<connectivity>(
        image, buffer, minimal_group_size,
        *label_trees16, is_background, false);
    }

    if (!success) {
      auto label_trees32 =
        dynamic_cast<imgproc_impl::EquivalenceLabelTrees<uint32_t> *>(label_trees_.get());

      if (!label_trees32) {
        label_trees_ = std::make_unique<imgproc_impl::EquivalenceLabelTrees<uint32_t>>();
        label_trees32 =
          dynamic_cast<imgproc_impl::EquivalenceLabelTrees<uint32_t> *>(label_trees_.get());
      }
      tryRemoveGroupsWithLabelType<connectivity>(
        image, buffer, minimal_group_size, *label_trees32,
        is_background, true);
    }
  }

  template<ConnectivityType connectivity, class Label, class IsBg>
  bool tryRemoveGroupsWithLabelType(
    Image<uint8_t> & image, MemoryBuffer & buffer, size_t minimal_group_size,
    imgproc_impl::EquivalenceLabelTrees<Label> & label_trees,
    const IsBg & is_background,
    bool throw_on_label_overflow) const
  {
    bool success{};
    try {
      removeGroupsImpl<connectivity>(image, buffer, label_trees, minimal_group_size, is_background);
      success = true;
    } catch (imgproc_impl::LabelOverflow &) {
      if (throw_on_label_overflow) {
        throw;
      }
    }
    return success;
  }

  template<ConnectivityType connectivity, class Label, class IsBg>
  void removeGroupsImpl(
    Image<uint8_t> & image, MemoryBuffer & buffer,
    imgproc_impl::EquivalenceLabelTrees<Label> & label_trees, size_t minimal_group_size,
    const IsBg & is_background) const
  {
    Label groups_count;
    auto labels = connectedComponents<connectivity>(
      image, buffer, label_trees,
      is_background, groups_count);

    const Label max_label_value = groups_count - 1;
    std::vector<size_t> groups_sizes = histogram(
      labels, max_label_value, size_t(minimal_group_size + 1));

    if (!groups_sizes.empty()) {
      groups_sizes.front() = 0;
    }

    std::vector<bool> noise_labels_table(groups_sizes.size());
    auto transform_fn = [&minimal_group_size](size_t bin_value) {
        return bin_value < minimal_group_size;
      };
    std::transform(
      groups_sizes.begin(), groups_sizes.end(), noise_labels_table.begin(),
      transform_fn);

    labels.convert(
      image, [&](Label src, uint8_t & trg) {
        if (!is_background(trg) && noise_labels_table[src]) {
          trg = 0;
        }
      });
  }

private:
  mutable std::unique_ptr<imgproc_impl::EquivalenceLabelTreesBase> label_trees_;
};

}  // namespace imgproc_impl

template<ConnectivityType connectivity, class Label, class IsBg>
Image<Label> connectedComponents(
  const Image<uint8_t> & image, MemoryBuffer & buffer,
  imgproc_impl::EquivalenceLabelTrees<Label> & label_trees,
  const IsBg & is_background,
  Label & total_labels)
{
  using namespace imgproc_impl;
  const size_t pixels = image.rows() * image.columns();

  if (pixels == 0) {
    total_labels = 0;
    return Image<Label>{};
  }

  Label * image_buffer = buffer.get<Label>(pixels);
  Image<Label> labels(image.rows(), image.columns(), image_buffer, image.columns());
  label_trees.reset(image.rows(), image.columns(), connectivity);
  total_labels = connectedComponentsImpl<connectivity>(
    image, labels, label_trees,
    is_background);
  return labels;
}

}  // namespace costmap_2d

#endif  // COSTMAP_2D__DENOISE__IMAGE_PROCESSING_HPP_
