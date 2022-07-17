#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  // The raw API - do not manage any dynamic memory
  template<typename T> LIBTENSOR_EXPORT void stack_outer_raw(size_t M, size_t N, T* output, const T* inputs[]);
  template<typename T> LIBTENSOR_EXPORT void stack_inner_raw(size_t M, size_t N, T* output, const T* inputs[]);

  // The API wraped in a Tensor type
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> stack_outer(std::vector<Tensor<const T>> values);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> stack_inner(std::vector<Tensor<const T>> values);
}

