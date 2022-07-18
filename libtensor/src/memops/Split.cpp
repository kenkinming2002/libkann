#include <libtensor/memops/Split.hpp>

namespace tensor
{
  template<typename T>
  std::vector<Tensor<T>> split_outer(Tensor<T> value)
  {
    const auto& shape = value.shape;
    const size_t M = shape.dimension(0);
    const size_t N = shape.drop_front(1).size();

    // Boilerplate code for type conversion for std vector
    auto results_buffer = std::vector<std::shared_ptr<Buffer<T>>>();
    for(size_t i=0; i<M; ++i)
      results_buffer.push_back(std::make_shared<Buffer<T>>(N));

    auto value_buffer = value.buffer;
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
      {
        const auto& result_buffer = results_buffer[m];
        (*result_buffer)[n] = (*value_buffer)[m*N+n];
      }

    auto results = std::vector<Tensor<T>>();
    for(size_t i=0; i<M; ++i)
      results.push_back(Tensor<T>(shape.drop_front(1), std::move(results_buffer[i])));
    return results;
  }

  template<typename T>
  std::vector<Tensor<T>> split_inner(Tensor<T> value)
  {
    const auto& shape = value.shape;
    const size_t M = shape.drop_back(1).size();
    const size_t N = shape.dimension(shape.rank() - 1);

    // Boilerplate code for type conversion for std vector
    auto results_buffer = std::vector<std::shared_ptr<Buffer<T>>>();
    for(size_t i=0; i<N; ++i)
      results_buffer.push_back(std::make_shared<Buffer<T>>(M));

    auto value_buffer = value.buffer;
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
      {
        const auto& result_buffer = results_buffer[n];
        (*result_buffer)[m] = (*value_buffer)[m*N+n];
      }

    auto results = std::vector<Tensor<T>>();
    for(size_t i=0; i<N; ++i)
      results.push_back(Tensor<T>(shape.drop_front(1), std::move(results_buffer[i])));
    return results;
  }

  template std::vector<Tensor<float>> split_outer(Tensor<float> value);
  template std::vector<Tensor<double>> split_outer(Tensor<double> value);
  template std::vector<Tensor<long double>> split_outer(Tensor<long double> value);

  template std::vector<Tensor<float>> split_inner(Tensor<float> value);
  template std::vector<Tensor<double>> split_inner(Tensor<double> value);
  template std::vector<Tensor<long double>> split_inner(Tensor<long double> value);
}
