#include <libtensor/MaxCoeff.hpp>

namespace tensor
{
  template<typename T>
  size_t max_coeff(Tensor<const T> value)
  {
    size_t coeff = std::numeric_limits<size_t>::max();
    T      max   = -std::numeric_limits<T>::infinity();

    for(size_t i=0; i<value.size(); ++i)
      if(max<value.data()[i])
      {
        coeff = i;
        max = value.data()[i];
      }

    return coeff;
  }

  template size_t max_coeff(Tensor<const float>       a);
  template size_t max_coeff(Tensor<const double>      a);
  template size_t max_coeff(Tensor<const long double> a);
}

