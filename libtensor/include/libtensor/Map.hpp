#pragma once

#include <libtensor/Tensor.hpp>
#include <utility>
#include <algorithm>

namespace tensor
{
  template<typename T> Tensor<T> unary_map(Tensor<T> a, const auto& f);
  template<typename T> Tensor<T> binary_map(Tensor<T> a, Tensor<T> b, const auto& f);
}

#include <assert.h>

namespace tensor
{
  template<typename T>
  Tensor<T> unary_map(Tensor<T> a, const auto& f)
  {
    auto buffer_a   = std::move(a.buffer);
    auto buffer_res = std::make_shared<Buffer<T>>(a.shape.size());

    for(size_t i=0; i<buffer_res->data().size(); ++i)
      buffer_res->data().data()[i] = f(buffer_a->data().data()[i]);

    return Tensor<T>(std::move(a.shape), std::move(buffer_res));
  }

  template<typename T>
  Tensor<T> binary_map(Tensor<T> a, Tensor<T> b, const auto& f)
  {
    assert(a.shape == b.shape);

    auto buffer_a   = std::move(a.buffer);
    auto buffer_b   = std::move(b.buffer);
    auto buffer_res = std::make_shared<Buffer<T>>(a.shape.size());

    for(size_t i=0; i<buffer_res->data().size(); ++i)
      buffer_res->data().data()[i] = f(buffer_a->data().data()[i], buffer_b->data().data()[i]);

    return Tensor<T>(std::move(a.shape), std::move(buffer_res));
  }
}
