#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT std::vector<Tensor<T>> split_outer(Tensor<T> value);
  template<typename T> LIBTENSOR_EXPORT std::vector<Tensor<T>> split_inner(Tensor<T> value);
}


