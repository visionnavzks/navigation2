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

#ifndef MY_COSTMAP_2D__DENOISE__IMAGE_HPP_
#define MY_COSTMAP_2D__DENOISE__IMAGE_HPP_

#include <cstddef>
#include <stdexcept>

namespace my_costmap_2d
{

template<class T>
class Image
{
public:
  Image() = default;

  Image(size_t rows, size_t columns, T * data, size_t step);

  Image(const Image & other);

  Image(Image && other) noexcept;

  size_t rows() const {return rows_;}
  size_t columns() const {return columns_;}
  bool empty() const {return rows_ == 0 || columns_ == 0;}
  size_t step() const {return step_;}

  T * row(size_t row);
  const T * row(size_t row) const;

  template<class Functor>
  void forEach(Functor && fn);

  template<class Functor>
  void forEach(Functor && fn) const;

  template<class TargetElement, class Converter>
  void convert(Image<TargetElement> & target, Converter && converter) const;

private:
  T * data_start_{};
  size_t rows_{};
  size_t columns_{};
  size_t step_{};
};

template<class T>
Image<T>::Image(size_t rows, size_t columns, T * data, size_t step)
: rows_{rows}, columns_{columns}, step_{step}
{
  data_start_ = data;
}

template<class T>
Image<T>::Image(const Image & other)
: data_start_{other.data_start_},
  rows_{other.rows_}, columns_{other.columns_}, step_{other.step_} {}

template<class T>
Image<T>::Image(Image && other) noexcept
: data_start_{other.data_start_},
  rows_{other.rows_}, columns_{other.columns_}, step_{other.step_} {}

template<class T>
T * Image<T>::row(size_t row)
{
  return const_cast<T *>( static_cast<const Image<T> &>(*this).row(row) );
}

template<class T>
const T * Image<T>::row(size_t row) const
{
  return data_start_ + row * step_;
}

template<class T>
template<class Functor>
void Image<T>::forEach(Functor && fn)
{
  static_cast<const Image<T> &>(*this).forEach(
    [&](const T & pixel) {
      fn(const_cast<T &>(pixel));
    });
}

template<class T>
template<class Functor>
void Image<T>::forEach(Functor && fn) const
{
  const T * rowPtr = row(0);

  for (size_t row = 0; row < rows(); ++row) {
    const T * rowEnd = rowPtr + columns();

    for (const T * pixel = rowPtr; pixel != rowEnd; ++pixel) {
      fn(*pixel);
    }
    rowPtr += step();
  }
}

template<class T>
template<class TargetElement, class Converter>
void Image<T>::convert(Image<TargetElement> & target, Converter && converter) const
{
  if (rows() != target.rows() || columns() != target.columns()) {
    throw std::logic_error("Image::convert. The source and target images size are different");
  }
  const T * source_row = row(0);
  TargetElement * target_row = target.row(0);

  for (size_t row = 0; row < rows(); ++row) {
    const T * rowInEnd = source_row + columns();
    const T * src = source_row;
    TargetElement * trg = target_row;

    for (; src != rowInEnd; ++src, ++trg) {
      converter(*src, *trg);
    }
    source_row += step();
    target_row += target.step();
  }
}

}  // namespace my_costmap_2d

#endif  // MY_COSTMAP_2D__DENOISE__IMAGE_HPP_
