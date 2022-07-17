#include <libkann/datasets/Random.hpp>

#include <libtensor/Tensor.hpp>

#include <range/v3/all.hpp>

#include <stdlib.h>

namespace kann
{
  std::vector<tensor::Tensor<float>> create_random_data(tensor::Shape shape, size_t count)
  {
    std::vector<tensor::Tensor<float>> results;
    results.reserve(count);
    for(size_t i=0; i<count; ++i)
    {
      auto buffer = std::make_shared<tensor::Buffer<float>>(shape.size());
      std::generate_n(buffer->data().data(), buffer->data().size(), [](){
        float tmp = (float)rand() / (float)RAND_MAX;
        return 2.0 * tmp - 1.0;
      });
      results.push_back(tensor::Tensor<float>(shape, std::move(buffer)));
    }
    return results;
  }
}

