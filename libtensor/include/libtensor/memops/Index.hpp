#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<T> index_outer(Tensor<T> value, std::vector<size_t> indices);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> index_inner(Tensor<T> value, std::vector<size_t> indices);
}



