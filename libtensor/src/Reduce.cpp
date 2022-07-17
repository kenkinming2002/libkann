#include <libtensor/Reduce.hpp>

#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<typename T>
  static inline void reduce_outer_raw(size_t M, size_t N, const T* __restrict__ input, T* __restrict__ output)
  {
    std::fill_n(output, N, T{});
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        output[n] += input[m*N+n];
  }

  template<typename T>
  static inline void reduce_inner_raw(size_t M, size_t N, const T* __restrict__ input, T* __restrict__ output)
  {
    std::fill_n(output, M, T{});
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        output[m] += input[m*N+n];
  }

  template<typename T>
  Tensor<T> reduce_outer(Tensor<T> value)
  {
    size_t M = value.shape.dimension(0);
    size_t N = value.shape.dimension(1);

    auto value_buffer  = value.buffer;
    auto result_buffer = std::make_shared<Buffer<T>>(N);
    reduce_outer_raw(M, N,
      value_buffer->data().data(),
      result_buffer->data().data()
    );
    return Tensor<T>(Shape::make(N), std::move(result_buffer));
  }

  template<typename T>
  Tensor<T> reduce_inner(Tensor<T> value)
  {
    size_t M = value.shape.dimension(0);
    size_t N = value.shape.dimension(1);

    auto value_buffer  = value.buffer;
    auto result_buffer = std::make_shared<Buffer<T>>(M);
    reduce_inner_raw(M, N,
      value_buffer->data().data(),
      result_buffer->data().data()
    );
    return Tensor<T>(Shape::make(M), std::move(result_buffer));
  }

  template Tensor<float> reduce_outer(Tensor<float> value);
  template Tensor<float> reduce_inner(Tensor<float> value);

  template Tensor<double> reduce_outer(Tensor<double> value);
  template Tensor<double> reduce_inner(Tensor<double> value);

  template Tensor<long double> reduce_outer(Tensor<long double> value);
  template Tensor<long double> reduce_inner(Tensor<long double> value);
}
