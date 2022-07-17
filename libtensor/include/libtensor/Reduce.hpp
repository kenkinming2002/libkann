#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<T> reduce_outer(Tensor<T> value);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> reduce_inner(Tensor<T> value);
}
