#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <SFML/Graphics/Image.hpp>

namespace kann
{
  KANN_EXPORT sf::Image toImage(TensorRef data);
}
