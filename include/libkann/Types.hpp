#pragma once

#include <memory>
#include <unordered_map>

namespace kann
{
  template<typename T>
  using CRef = std::shared_ptr<const T>;

  template<typename T>
  using Ref = std::shared_ptr<T>;
}
