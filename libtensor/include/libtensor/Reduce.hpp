#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Direction.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<T> reduce_outer(Tensor<const T> value);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> reduce_inner(Tensor<const T> value);
}
