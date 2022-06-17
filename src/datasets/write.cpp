#include <libkann/datasets/write.hpp>

#include <libkann/Tensor.hpp>

namespace kann
{
  sf::Image toImage(TensorRef data)
  {
    size_t height = data.shape().dimension(0);
    size_t width  = data.shape().dimension(1);

    std::vector<sf::Uint8> pixels;
    pixels.reserve(width * height * 4);
    for(size_t i=0; i<width*height; ++i)
    {
      const auto pixel = static_cast<sf::Uint8>(data.get(i) * 256.0);
      for(size_t j=0; j<3; ++j)
        pixels.push_back(pixel);

      // No transparency
      pixels.push_back(255);
    }

    sf::Image image;
    image.create(width, height, pixels.data());
    return image;
  }
}
