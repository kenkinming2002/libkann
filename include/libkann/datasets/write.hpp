#pragma once

#include <Eigen/Eigen>

#include <SFML/Graphics/Image.hpp>

#include <ostream>

namespace kann
{
  sf::Image toImage(const Eigen::VectorXd& data, size_t width, size_t height);
}
