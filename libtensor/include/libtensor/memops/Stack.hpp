#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<T> stack_outer(std::vector<Tensor<T>> values);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> stack_inner(std::vector<Tensor<T>> values);
}

