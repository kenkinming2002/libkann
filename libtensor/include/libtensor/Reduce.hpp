#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Direction.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<Direction direction, typename T> Tensor<T> reduce(Tensor<const T> value);

  extern template LIBTENSOR_EXPORT Tensor<float> reduce<Direction::LEFT> (Tensor<const float> value);
  extern template LIBTENSOR_EXPORT Tensor<float> reduce<Direction::RIGHT>(Tensor<const float> value);

  extern template LIBTENSOR_EXPORT Tensor<double> reduce<Direction::LEFT> (Tensor<const double> value);
  extern template LIBTENSOR_EXPORT Tensor<double> reduce<Direction::RIGHT>(Tensor<const double> value);

  extern template LIBTENSOR_EXPORT Tensor<long double> reduce<Direction::LEFT> (Tensor<const long double> value);
  extern template LIBTENSOR_EXPORT Tensor<long double> reduce<Direction::RIGHT>(Tensor<const long double> value);
}

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
  Tensor<T> reduce(Tensor<const T> value)
  {
    auto [M, N] = std::make_pair(value.dimension(0), value.dimension(1));
    if constexpr(direction == Direction::LEFT)
    {
      auto result = Tensor<T>::create(Shape(N));
      to_array1d(result) = to_array2d(value).colwise().sum();
      return result;
    }
    else if constexpr(direction == Direction::RIGHT)
    {
      auto result = Tensor<T>::create(Shape(M));
      to_array1d(result) = to_array2d(value).rowwise().sum();
      return result;
    }
    else
      static_assert("Invalid direction");
  }
}
