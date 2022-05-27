#pragma once

#include <libkann/Types.hpp>

#include <SFML/Graphics/Image.hpp>

namespace kann
{
  sf::Image toImage(const Tensor& data, size_t width, size_t height);
}
