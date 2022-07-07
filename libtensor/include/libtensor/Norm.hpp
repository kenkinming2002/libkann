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

// Implementation
namespace tensor
{
  template<typename T>
  T norm(Tensor<const T> value)
  {
    T sum{};
    for(size_t i=0; i<value.size(); ++i)
      sum += value.data()[i] * value.data()[i];
    return std::sqrt(sum);
  }
}
