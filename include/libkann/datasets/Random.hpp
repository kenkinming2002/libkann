#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Shape.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<Tensor> create_random_data(Shape shape, size_t count);
}


