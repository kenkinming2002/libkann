#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Direction.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<Direction direction, typename T> LIBTENSOR_EXPORT Tensor<T> reduce(Tensor<const T> value);

  extern template LIBTENSOR_EXPORT Tensor<float> reduce<Direction::LEFT> (Tensor<const float> value);
  extern template LIBTENSOR_EXPORT Tensor<float> reduce<Direction::RIGHT>(Tensor<const float> value);

  extern template LIBTENSOR_EXPORT Tensor<double> reduce<Direction::LEFT> (Tensor<const double> value);
  extern template LIBTENSOR_EXPORT Tensor<double> reduce<Direction::RIGHT>(Tensor<const double> value);

  extern template LIBTENSOR_EXPORT Tensor<long double> reduce<Direction::LEFT> (Tensor<const long double> value);
  extern template LIBTENSOR_EXPORT Tensor<long double> reduce<Direction::RIGHT>(Tensor<const long double> value);
}
