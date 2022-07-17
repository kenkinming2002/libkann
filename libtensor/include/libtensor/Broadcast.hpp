#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

// Declaration
namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_add_outer(Tensor<T> a, Tensor<T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_sub_outer(Tensor<T> a, Tensor<T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_mul_outer(Tensor<T> a, Tensor<T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_div_outer(Tensor<T> a, Tensor<T> b);

  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_add_inner(Tensor<T> a, Tensor<T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_sub_inner(Tensor<T> a, Tensor<T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_mul_inner(Tensor<T> a, Tensor<T> b);
  template<typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_div_inner(Tensor<T> a, Tensor<T> b);
}
