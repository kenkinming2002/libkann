#pragma once

#include <libkann/Tensor.hpp>

#include <SFML/Graphics/Image.hpp>

#include <ostream>

namespace kann
{
  sf::Image toImage(const Tensor& data, size_t width, size_t height);
}
