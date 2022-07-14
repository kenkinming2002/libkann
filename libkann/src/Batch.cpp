#include <libkann/Batch.hpp>

#include <range/v3/all.hpp>
#include <range/v3/view/join.hpp>

namespace kann
{
  std::vector<tensor::Tensor<const float>> batch(std::vector<tensor::Tensor<const float>> values, size_t batch_size)
  {
    std::vector<tensor::Tensor<const float>> results;

    const tensor::Shape value_shape  = values.front().shape();
    const tensor::Shape result_shape = tensor::Shape::concat(std::array{tensor::Shape(batch_size), value_shape});
    const size_t size = value_shape.size();
    for(size_t i=0; i+batch_size<=values.size(); i+=batch_size)
    {
      tensor::Tensor<float> result = tensor::Tensor<float>::create(result_shape);
      for(size_t k=0; k<batch_size; ++k)
      {
        tensor::Tensor<const float> value = values[i+k];
        std::copy_n(value.data(), size, result.data() + k * size);
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  std::vector<tensor::Tensor<const float>> unbatch(std::vector<tensor::Tensor<const float>> values, size_t batch_size)
  {
    std::vector<tensor::Tensor<const float>> results;

    const tensor::Shape value_shape  = values.front().shape();
    const tensor::Shape result_shape = value_shape.drop_front(1);
    const size_t size = result_shape.size();
    assert(value_shape.dimension(0) == batch_size);
    for(size_t i=0; i<values.size(); ++i)
    {
      tensor::Tensor<const float> value = values[i];
      for(size_t k=0; k<batch_size; ++k)
      {
        tensor::Tensor<float> result = tensor::Tensor<float>::create(result_shape);
        std::copy_n(value.data() + k * size, size, result.data());
        results.push_back(std::move(result));
      }
    }
    return results;
  }
}

