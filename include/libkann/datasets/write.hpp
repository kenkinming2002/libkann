#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <SFML/Graphics/Image.hpp>

namespace kann
{
  sf::Image toImage(TensorRef data);
}
