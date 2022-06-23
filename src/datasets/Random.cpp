#include <libkann/datasets/Random.hpp>

#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

#include <stdlib.h>

namespace kann
{
  std::vector<Tensor<float>> create_random_data(Shape shape, size_t count)
  {
    return ranges::views::generate_n([&]() {
      Tensor<float> result = Tensor<float>::create(shape);
      ranges::generate_n(result.as_ref().data(), result.as_ref().size(), []() {
        float tmp = (float)rand() / RAND_MAX;
        return 2.0 * tmp - 1.0;
      });
      return result;
    }, count) | ranges::to_vector;
  }
}

