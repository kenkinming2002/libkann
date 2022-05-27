#include <libkann/datasets/Random.hpp>

#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  std::vector<tensor_t> create_random_data(size_t size, size_t count)
  {
    return ranges::views::generate_n([size]() -> tensor_t {
      auto result = std::make_shared<Tensor>(size);
      result->asArray().setRandom();
      return result;
    }, count) | ranges::to_vector;
  }
}

