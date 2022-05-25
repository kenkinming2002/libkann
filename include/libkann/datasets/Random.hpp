#pragma once

#include <vector>
#include <memory>

namespace kann
{
  struct Tensor;

  std::vector<std::shared_ptr<const Tensor>> create_random_data(size_t size, size_t count);
}


