// Copyright (c) 2021 RoboTech Vision
// Copyright (c) 2020, Samsung Research America
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

#ifndef CONSTRAINED_SMOOTHER__SMOOTHER_RUN_BASE_HPP_
#define CONSTRAINED_SMOOTHER__SMOOTHER_RUN_BASE_HPP_

namespace constrained_smoother
{

/// 复用两种 smoother 的单次执行骨架。
///
/// Derived 只需要实现 prepare()、solve()、finalize() 三段，基类负责把它们
/// 串成统一的执行主线，并保留 owner / request 两个最小共享上下文。
template<typename Derived, typename Owner, typename Request>
class SmootherRunBase
{
public:
  bool execute()
  {
    Derived & derived = static_cast<Derived &>(*this);
    derived.prepare();
    if (!derived.solve()) {
      return false;
    }
    return derived.finalize();
  }

protected:
  SmootherRunBase(Owner & owner, const Request & request)
  : owner_(owner), request_(request)
  {
  }

  Owner & owner()
  {
    return owner_;
  }

  const Owner & owner() const
  {
    return owner_;
  }

  const Request & request() const
  {
    return request_;
  }

private:
  Owner & owner_;
  const Request & request_;
};

}  // namespace constrained_smoother

#endif  // CONSTRAINED_SMOOTHER__SMOOTHER_RUN_BASE_HPP_