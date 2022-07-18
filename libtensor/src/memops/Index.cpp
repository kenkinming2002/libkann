#include <libtensor/memops/Index.hpp>

#include <libtensor/memops/Split.hpp>
#include <libtensor/memops/Stack.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> index_outer(Tensor<T> value, std::vector<size_t> indices)
  {
    const auto& old_shape = value.shape;
    const size_t old_M = old_shape.dimension(0);

    const auto& sub_shape = old_shape.drop_front(1);
    const size_t N = sub_shape.size();

    const size_t new_M = indices.size();
    const auto& new_shape = Shape::make(new_M, sub_shape);

    auto value_buffer  = value.buffer;
    auto result_buffer = std::make_shared<Buffer<T>>(new_M * N);
    for(size_t new_m = 0; new_m<new_M; ++new_m)
    {
      const size_t old_m = indices[new_m];
      assert(old_m<old_M);
      for(size_t n=0; n<N; ++n)
        (*result_buffer)[new_m*N+n] = (*value_buffer)[old_m*N+n];
    }

    return Tensor<T>(std::move(new_shape), std::move(result_buffer));
  }

  template<typename T>
  Tensor<T> index_inner(Tensor<T> value, std::vector<size_t> indices)
  {
    const auto& old_shape = value.shape;
    const size_t old_N = old_shape.dimension(old_shape.rank()-1);

    const auto& sub_shape = old_shape.drop_back(1);
    const size_t M = sub_shape.size();

    const size_t new_N = indices.size();
    const auto& new_shape = Shape::make(sub_shape, new_N);

    auto value_buffer  = value.buffer;
    auto result_buffer = std::make_shared<Buffer<T>>(M * new_N);
    for(size_t m = 0; m<M; ++m)
      for(size_t new_n=0; new_n<new_N; ++new_n)
      {
        const size_t old_n = indices[new_n];
        assert(old_n<old_N);
        (*result_buffer)[m*new_N+new_n] = (*value_buffer)[m*old_N+old_n];
      }

    return Tensor<T>(std::move(new_shape), std::move(result_buffer));
  }

  template Tensor<float> index_outer(Tensor<float> value, std::vector<size_t> indices);
  template Tensor<double> index_outer(Tensor<double> value, std::vector<size_t> indices);
  template Tensor<long double> index_outer(Tensor<long double> value, std::vector<size_t> indices);

  template Tensor<float> index_inner(Tensor<float> value, std::vector<size_t> indices);
  template Tensor<double> index_inner(Tensor<double> value, std::vector<size_t> indices);
  template Tensor<long double> index_inner(Tensor<long double> value, std::vector<size_t> indices);
}
