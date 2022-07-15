#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Direction.hpp>
#include <libtensor/Tensor.hpp>

// Declaration
namespace tensor
{
  template<Direction direction, typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_add(Tensor<const T> a, Tensor<const T> b);
  template<Direction direction, typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_sub(Tensor<const T> a, Tensor<const T> b);
  template<Direction direction, typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_mul(Tensor<const T> a, Tensor<const T> b);
  template<Direction direction, typename T> LIBTENSOR_EXPORT Tensor<T> broadcast_div(Tensor<const T> a, Tensor<const T> b);

  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_add<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);
  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_sub<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);
  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_mul<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);
  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_div<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);

  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_add<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);
  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_sub<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);
  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_mul<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);
  extern template LIBTENSOR_EXPORT Tensor<float> broadcast_div<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);

  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_add<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);
  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_sub<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);
  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_mul<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);
  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_div<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);

  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_add<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);
  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_sub<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);
  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_mul<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);
  extern template LIBTENSOR_EXPORT Tensor<double> broadcast_div<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);

  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_add<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);
  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_sub<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);
  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_mul<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);
  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_div<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);

  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_add<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_sub<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_mul<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
  extern template LIBTENSOR_EXPORT Tensor<long double> broadcast_div<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
}
