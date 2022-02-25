#pragma once

#include <libkann/QualifiedName.hpp>

#include <memory>
#include <unordered_map>

namespace kann
{
  template<typename T>
  using CRef = std::shared_ptr<const T>;

  template<typename T>
  using Ref = std::shared_ptr<T>;

  template<typename T>
  using Map = std::unordered_map<QualifiedName, CRef<T>>;
}
