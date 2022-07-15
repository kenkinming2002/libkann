#include <libtensor/Norm.hpp>

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

  template float       norm(Tensor<const float>       a);
  template double      norm(Tensor<const double>      a);
  template long double norm(Tensor<const long double> a);
}
