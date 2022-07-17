#include <libtensor/Initializer.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> create_constant(tensor::Shape shape, T value)
  {
    auto buffer = std::make_shared<Buffer<T>>(shape.size());
    std::fill_n(buffer->data().data(), buffer->data().size(), value);
    return Tensor<T>(std::move(shape), std::move(buffer));
  }

  template<typename T>
  Tensor<T> create_normal(tensor::Shape shape, T mean, T stddev, std::default_random_engine& prng)
  {
    auto buffer = std::make_shared<Buffer<T>>(shape.size());
    std::normal_distribution<T> dist(mean, stddev);
    std::generate_n(buffer->data().data(), buffer->data().size(),  [&](){ return dist(prng); });
    return Tensor<T>(std::move(shape), std::move(buffer));
  }

  template<typename T>
  Tensor<T> create_uniform(tensor::Shape shape, T a, T b, std::default_random_engine& prng)
  {
    auto buffer = std::make_shared<Buffer<T>>(shape.size());
    std::uniform_real_distribution<T> dist(a, b);
    std::generate_n(buffer->data().data(), buffer->data().size(),  [&](){ return dist(prng); });
    return Tensor<T>(std::move(shape), std::move(buffer));
  }

  template Tensor<float> create_constant(tensor::Shape shape, float value);
  template Tensor<float> create_normal(tensor::Shape shape, float mean, float stddev, std::default_random_engine& prng);
  template Tensor<float> create_uniform(tensor::Shape shape, float a, float b, std::default_random_engine& prng);

  template Tensor<double> create_constant(tensor::Shape shape, double value);
  template Tensor<double> create_normal(tensor::Shape shape, double mean, double stddev, std::default_random_engine& prng);
  template Tensor<double> create_uniform(tensor::Shape shape, double a, double b, std::default_random_engine& prng);

  template Tensor<long double> create_constant(tensor::Shape shape, long double value);
  template Tensor<long double> create_normal(tensor::Shape shape, long double mean, long double stddev, std::default_random_engine& prng);
  template Tensor<long double> create_uniform(tensor::Shape shape, long double a, long double b, std::default_random_engine& prng);
}
