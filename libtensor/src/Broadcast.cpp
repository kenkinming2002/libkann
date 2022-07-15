#include <libtensor/Broadcast.hpp>

#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> broadcast_add_outer(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).rowwise() + details::to_array1d(b);
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_sub_outer(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).rowwise() - details::to_array1d(b);
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_mul_outer(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).rowwise() * details::to_array1d(b);
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_div_outer(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).rowwise() / details::to_array1d(b);
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_add_inner(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).colwise() + details::to_array1d(b).transpose();
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_sub_inner(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).colwise() - details::to_array1d(b).transpose();
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_mul_inner(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).colwise() * details::to_array1d(b).transpose();
    return c;
  }

  template<typename T>
  Tensor<T> broadcast_div_inner(Tensor<const T> a, Tensor<const T> b)
  {
    auto c = Tensor<T>::create(a.shape());
    details::to_array2d(c) = details::to_array2d(a).colwise() / details::to_array1d(b).transpose();
    return c;
  }

  template Tensor<float> broadcast_add_outer(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_sub_outer(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_mul_outer(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_div_outer(Tensor<const float> a, Tensor<const float> b);

  template Tensor<float> broadcast_add_inner(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_sub_inner(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_mul_inner(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_div_inner(Tensor<const float> a, Tensor<const float> b);

  template Tensor<double> broadcast_add_outer(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_sub_outer(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_mul_outer(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_div_outer(Tensor<const double> a, Tensor<const double> b);

  template Tensor<double> broadcast_add_inner(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_sub_inner(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_mul_inner(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_div_inner(Tensor<const double> a, Tensor<const double> b);

  template Tensor<long double> broadcast_add_outer(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_sub_outer(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_mul_outer(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_div_outer(Tensor<const long double> a, Tensor<const long double> b);

  template Tensor<long double> broadcast_add_inner(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_sub_inner(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_mul_inner(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_div_inner(Tensor<const long double> a, Tensor<const long double> b);
}
