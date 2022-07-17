#include <libkann/Batch.hpp>

#include <libtensor/Stack.hpp>

#include <range/v3/all.hpp>
#include <range/v3/view/join.hpp>

namespace kann
{
  std::vector<tensor::Tensor<float>> batch(std::vector<tensor::Tensor<float>> values, size_t batch_size)
  {
    std::vector<tensor::Tensor<float>> results;
    for(size_t i=0; i+batch_size<=values.size(); i+=batch_size)
      results.push_back(tensor::stack_outer(std::vector(&values[i], &values[i+batch_size])));

    return results;
  }

  std::vector<tensor::Tensor<float>> unbatch(std::vector<tensor::Tensor<float>> values, size_t batch_size)
  {
    std::vector<tensor::Tensor<float>> results;

    const tensor::Shape value_shape  = values.front().shape;
    const tensor::Shape result_shape = value_shape.drop_front(1);
    const size_t size = result_shape.size();
    assert(value_shape.dimension(0) == batch_size);
    for(size_t i=0; i<values.size(); ++i)
    {
      auto value = values[i];
      auto value_buffer  = value.buffer;
      for(size_t k=0; k<batch_size; ++k)
      {
        auto result_buffer = std::make_shared<tensor::Buffer<float>>(result_shape.size());
        std::copy_n(
          value_buffer->data().data() + k * size,
          size,
          result_buffer->data().data()
        );
        results.push_back(tensor::Tensor<float>(result_shape, std::move(result_buffer)));
      }
    }
    return results;
  }
}

