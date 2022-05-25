#include <libkann/datasets/Random.hpp>

#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  std::vector<std::shared_ptr<const Tensor>> create_random_data(size_t size, size_t count)
  {
    return ranges::views::generate_n([size]() -> std::shared_ptr<const Tensor> {
      auto result = std::make_shared<Tensor>(size);
      result->asArray().setRandom();
      return result;
    }, count) | ranges::to_vector;
  }
}

