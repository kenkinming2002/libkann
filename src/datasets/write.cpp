#include <libkann/datasets/write.hpp>

namespace kann
{
  sf::Image toImage(const Eigen::VectorXd& data, size_t width, size_t height)
  {
    std::vector<sf::Uint8> pixels;
    pixels.reserve(width * height * 4);
    for(size_t i=0; i<width*height; ++i)
    {
      const auto pixel = static_cast<sf::Uint8>(data(i) * 256.0);
      for(size_t j=0; j<4; ++j)
        pixels.push_back(pixel);
    }

    sf::Image image;
    image.create(width, height, pixels.data());
    return image;
  }
}
