#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

// Declaration
namespace tensor
{
  enum class Direction { LEFT, RIGHT };

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
#include <type_traits>
#include <Eigen/Eigen>

namespace tensor
{
  template<typename T>
  static inline auto to_array1d(Tensor<T> value)
  {
    using ArrayType = std::conditional_t<std::is_const_v<T>,
      Eigen::Array<std::remove_const_t<T>, 1, Eigen::Dynamic> const,
      Eigen::Array<std::remove_const_t<T>, 1, Eigen::Dynamic>>;

    assert(value.rank() == 1);
    return ArrayType::Map(value.data(), value.dimension(0));
  }

  template<typename T>
  static inline auto to_array2d(Tensor<T> value)
  {
    using ArrayType = std::conditional_t<std::is_const_v<T>,
      Eigen::Array<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic> const,
      Eigen::Array<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic>>;

    assert(value.rank() == 2);
    return ArrayType::Map(value.data(), value.dimension(0), value.dimension(1));
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_add(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       to_array2d(c) = to_array2d(a).rowwise() + to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) to_array2d(c) = to_array2d(a).colwise() + to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_sub(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       to_array2d(c) = to_array2d(a).rowwise() - to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) to_array2d(c) = to_array2d(a).colwise() - to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_mul(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       to_array2d(c) = to_array2d(a).rowwise() * to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) to_array2d(c) = to_array2d(a).colwise() * to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }

  template<Direction direction, typename T>
  Tensor<T> broadcast_div(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    if constexpr(direction == Direction::LEFT)       to_array2d(c) = to_array2d(a).rowwise() / to_array1d(b);
    else if constexpr(direction == Direction::RIGHT) to_array2d(c) = to_array2d(a).colwise() / to_array1d(b).transpose();
    else static_assert("Invalid direction");
    return c;
  }
}
