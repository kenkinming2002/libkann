#include <libtensor/MaxCoeff.hpp>

namespace tensor
{
  template<typename T>
  size_t max_coeff(Tensor<T> value) noexcept
  {
    size_t coeff = std::numeric_limits<size_t>::max();
    T      max   = -std::numeric_limits<T>::infinity();

    size_t i = 0;
    for(T v : value.buffer->data())
    {
      if(max<v)
      {
        coeff = i;
        max   = v;
      }
      ++i;
    }

    return coeff;
  }

  template size_t max_coeff(Tensor<float>       a);
  template size_t max_coeff(Tensor<double>      a);
  template size_t max_coeff(Tensor<long double> a);
}

