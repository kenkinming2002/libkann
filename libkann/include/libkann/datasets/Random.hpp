#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Shape.hpp>
#include <libtensor/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<tensor::Tensor<float>> create_random_data(tensor::Shape shape, size_t count);
}


