#pragma once

// Declaration
#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <cmath>

namespace tensor
{
  template<typename T>
  T norm(Tensor<const T> a);

  extern template LIBTENSOR_EXPORT float       norm(Tensor<const float>       value);
  extern template LIBTENSOR_EXPORT double      norm(Tensor<const double>      value);
  extern template LIBTENSOR_EXPORT long double norm(Tensor<const long double> value);
}
