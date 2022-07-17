#include <libtensor/Norm.hpp>

namespace tensor
{
  template<typename T>
  T norm(Tensor<T> value)
  {
    T sum{};
    for(T v : value.buffer->data())
      sum += v * v;
    return std::sqrt(sum);
  }

  template float       norm(Tensor<float>       a);
  template double      norm(Tensor<double>      a);
  template long double norm(Tensor<long double> a);
}
