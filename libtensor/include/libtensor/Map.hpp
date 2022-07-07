#pragma once

#include <libtensor/Tensor.hpp>
#include <algorithm>

namespace tensor
{
  template<typename T> Tensor<T> unary_map(Tensor<const T> a, const auto& f);
  template<typename T> Tensor<T> binary_map(Tensor<const T> a, Tensor<const T> b, const auto& f);
}

#include <assert.h>

namespace tensor
{
  template<typename T>
  Tensor<T> unary_map(Tensor<const T> a, const auto& f)
  {
    auto result = Tensor<T>::create(a.shape());
    for(size_t i=0; i<a.size(); ++i)
      result.data()[i] = f(a.data()[i]);
    return result;
  }

  template<typename T>
  Tensor<T> binary_map(Tensor<const T> a, Tensor<const T> b, const auto& f)
  {
    assert(a.shape() == b.shape());

    auto result = Tensor<T>::create(a.shape());
    for(size_t i=0; i<a.size(); ++i)
      result.data()[i] = f(a.data()[i], b.data()[i]);
    return result;
  }
}
