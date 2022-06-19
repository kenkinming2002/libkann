#include <libkann/datasets/Random.hpp>

#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

#include <stdlib.h>

namespace kann
{
  std::vector<Tensor> create_random_data(Shape shape, size_t count)
  {
    return ranges::views::generate_n([&]() {
      MutableTensor result = MutableTensor::create(shape);
      ranges::generate_n(result.data(), result.size(), []() {
        float tmp = (float)rand() / RAND_MAX;
        return 2.0 * tmp - 1.0;
      });
      return std::move(result).as_const();
    }, count) | ranges::to_vector;
  }
}

