#include <libtensor/Reduce.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> reduce_outer(Tensor<T> value, ssize_t count)
  {
    if(count<0)
      count += value.shape.rank();

    assert(count>=0 && count<value.shape.rank());
    const Shape outer_shape = value.shape.front(count);
    const Shape inner_shape = value.shape.drop_front(count);

    const size_t M = outer_shape.size();
    const size_t N = inner_shape.size();

    auto value_buffer  = value.buffer;
    auto result_buffer = std::make_shared<Buffer<T>>(N);

    std::fill_n(result_buffer->data().data(), result_buffer->data().size(), T{});
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        (*result_buffer)[n] += (*value_buffer)[m*N+n];

    return Tensor<T>(inner_shape, std::move(result_buffer));
  }

  template<typename T>
  Tensor<T> reduce_inner(Tensor<T> value, ssize_t count)
  {
    if(count<0)
      count += value.shape.rank();

    assert(count>=0 && count<value.shape.rank());
    const Shape outer_shape = value.shape.drop_back(count);
    const Shape inner_shape = value.shape.back(count);

    const size_t M = outer_shape.size();
    const size_t N = inner_shape.size();

    auto value_buffer  = value.buffer;
    auto result_buffer = std::make_shared<Buffer<T>>(M);

    std::fill_n(result_buffer->data().data(), result_buffer->data().size(), T{});
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        (*result_buffer)[m] += (*value_buffer)[m*N+n];

    return Tensor<T>(outer_shape, std::move(result_buffer));
  }

  template Tensor<float> reduce_outer(Tensor<float> value, ssize_t count);
  template Tensor<float> reduce_inner(Tensor<float> value, ssize_t count);

  template Tensor<double> reduce_outer(Tensor<double> value, ssize_t count);
  template Tensor<double> reduce_inner(Tensor<double> value, ssize_t count);

  template Tensor<long double> reduce_outer(Tensor<long double> value, ssize_t count);
  template Tensor<long double> reduce_inner(Tensor<long double> value, ssize_t count);
}
