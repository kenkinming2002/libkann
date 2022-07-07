#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Direction.hpp>
#include <libtensor/Tensor.hpp>

// Declaration
namespace tensor
{
  template<Direction direction, typename T> Tensor<T> broadcast_add(Tensor<const T> a, Tensor<const T> b);
  template<Direction direction, typename T> Tensor<T> broadcast_sub(Tensor<const T> a, Tensor<const T> b);
  template<Direction direction, typename T> Tensor<T> broadcast_mul(Tensor<const T> a, Tensor<const T> b);
  template<Direction direction, typename T> Tensor<T> broadcast_div(Tensor<const T> a, Tensor<const T> b);

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

// Implementation
#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<Direction direction, typename T>
  Tensor<T> broadcast_add(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       details::to_array2d(c) = details::to_array2d(a).rowwise() + details::to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) details::to_array2d(c) = details::to_array2d(a).colwise() + details::to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_sub(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       details::to_array2d(c) = details::to_array2d(a).rowwise() - details::to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) details::to_array2d(c) = details::to_array2d(a).colwise() - details::to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_mul(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       details::to_array2d(c) = details::to_array2d(a).rowwise() * details::to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) details::to_array2d(c) = details::to_array2d(a).colwise() * details::to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_div(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       details::to_array2d(c) = details::to_array2d(a).rowwise() / details::to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) details::to_array2d(c) = details::to_array2d(a).colwise() / details::to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }
}
