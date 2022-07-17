#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T>
  LIBTENSOR_EXPORT Tensor<T> matrix_product(Tensor<T> a, bool trans_a, Tensor<T> b, bool trans_b);
}

