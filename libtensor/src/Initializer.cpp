#include <libtensor/Initializer.hpp>

namespace tensor
{
  template<typename T>
  Tensor<const T> create_constant(tensor::Shape shape, T value)
  {
    auto result = Tensor<T>::create(std::move(shape));
    std::fill_n(result.data(), result.size(), value);
    return result;
  }

  template<typename T>
  Tensor<const T> create_normal(tensor::Shape shape, T mean, T stddev, std::default_random_engine& prng)
  {
    auto result = Tensor<T>::create(std::move(shape));
    std::normal_distribution<T> dist(mean, stddev);
    std::generate_n(result.data(), result.size(), [&](){ return dist(prng); });
    return result;
  }

  template<typename T>
  Tensor<const T> create_uniform(tensor::Shape shape, T a, T b, std::default_random_engine& prng)
  {
    auto result = Tensor<T>::create(std::move(shape));
    std::uniform_real_distribution<T> dist(a, b);
    std::generate_n(result.data(), result.size(), [&](){ return dist(prng); });
    return result;
  }

  template Tensor<const float> create_constant(tensor::Shape shape, float value);
  template Tensor<const float> create_normal(tensor::Shape shape, float mean, float stddev, std::default_random_engine& prng);
  template Tensor<const float> create_uniform(tensor::Shape shape, float a, float b, std::default_random_engine& prng);

  template Tensor<const double> create_constant(tensor::Shape shape, double value);
  template Tensor<const double> create_normal(tensor::Shape shape, double mean, double stddev, std::default_random_engine& prng);
  template Tensor<const double> create_uniform(tensor::Shape shape, double a, double b, std::default_random_engine& prng);

  template Tensor<const long double> create_constant(tensor::Shape shape, long double value);
  template Tensor<const long double> create_normal(tensor::Shape shape, long double mean, long double stddev, std::default_random_engine& prng);
  template Tensor<const long double> create_uniform(tensor::Shape shape, long double a, long double b, std::default_random_engine& prng);
}
