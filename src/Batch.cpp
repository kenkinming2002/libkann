#include <libkann/Batch.hpp>

#include <range/v3/all.hpp>
#include <range/v3/view/join.hpp>

namespace kann
{
  std::vector<Tensor<float>> batch(const std::vector<Tensor<float>>& values, size_t batch_size)
  {
    const Shape shape = Shape::concat(Shape(batch_size), values.front().as_ref().shape());
    return values
      | ranges::views::take_exactly(values.size() / batch_size * batch_size) // Ensure that the each chunk is full
      | ranges::views::chunk(batch_size)
      | ranges::views::transform([&](auto&& sub_values){
          Tensor<float> result = Tensor<float>::create(shape);
          for(auto&& [i, value] : ranges::views::enumerate(std::forward<decltype(sub_values)>(sub_values)))
            std::copy_n(value.as_ref().data(), value.as_ref().size(), result.as_ref()[i].data());

          return result;
        })
      | ranges::to_vector;
  }

  std::vector<Tensor<float>> unbatch(const std::vector<Tensor<float>>& values, size_t batch_size)
  {
    std::vector<Tensor<float>> results;
    results.reserve(values.size() * batch_size);
    for(const Tensor<float>& value : values)
      for(size_t i=0; i<batch_size; ++i)
      {
        TensorRef<const float> value_slice = value.as_const_ref()[i];

        Tensor<float> result = Tensor<float>::create(value_slice.shape());
        std::copy_n(value_slice.data(), value_slice.size(), result.as_ref().data());
        results.push_back(std::move(result));
      }

    return results;
  }
}

