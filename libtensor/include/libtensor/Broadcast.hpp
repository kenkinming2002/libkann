#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Direction.hpp>
#include <libtensor/Tensor.hpp>

// Declaration
namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_add_outer(Tensor<const T> a, Tensor<const T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_sub_outer(Tensor<const T> a, Tensor<const T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_mul_outer(Tensor<const T> a, Tensor<const T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_div_outer(Tensor<const T> a, Tensor<const T> b);

  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_add_inner(Tensor<const T> a, Tensor<const T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_sub_inner(Tensor<const T> a, Tensor<const T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_mul_inner(Tensor<const T> a, Tensor<const T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> broadcast_div_inner(Tensor<const T> a, Tensor<const T> b);
}
