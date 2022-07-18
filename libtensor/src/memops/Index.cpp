#include <libtensor/memops/Index.hpp>

#include <libtensor/memops/Split.hpp>
#include <libtensor/memops/Stack.hpp>

namespace tensor
{
  // Lets do it inefficiently first

  template<typename T>
  Tensor<T> index_outer(Tensor<T> value, std::vector<size_t> indices)
  {
    auto values = split_outer(value);
    auto new_values = std::vector<Tensor<T>>();

    new_values.reserve(indices.size());
    for(size_t index : indices)
      new_values.push_back(std::move(values[index]));

    return stack_outer(std::move(new_values));
  }

  template<typename T>
  Tensor<T> index_inner(Tensor<T> value, std::vector<size_t> indices)
  {
    auto values = split_inner(value);
    auto new_values = std::vector<Tensor<T>>();

    new_values.reserve(indices.size());
    for(size_t index : indices)
      new_values.push_back(std::move(values[index]));

    return stack_inner(std::move(new_values));
  }

  template Tensor<float> index_outer(Tensor<float> value, std::vector<size_t> indices);
  template Tensor<double> index_outer(Tensor<double> value, std::vector<size_t> indices);
  template Tensor<long double> index_outer(Tensor<long double> value, std::vector<size_t> indices);

  template Tensor<float> index_inner(Tensor<float> value, std::vector<size_t> indices);
  template Tensor<double> index_inner(Tensor<double> value, std::vector<size_t> indices);
  template Tensor<long double> index_inner(Tensor<long double> value, std::vector<size_t> indices);
}
