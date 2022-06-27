#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <SFML/Graphics/Image.hpp>

namespace kann
{
  KANN_EXPORT sf::Image toImage(Tensor<const float> data);
}
