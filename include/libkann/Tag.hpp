#pragma once

#include <type_traits>
#include <limits>

namespace kann
{
  enum class Tag : unsigned
  {
    DEFAULT           = 1u << 0,
    ENCODER           = 1u << 1,
    DECODER           = 1u << 2,
    GAN_GENERATOR     = 1u << 3,
    GAN_DISCRIMINATOR = 1u << 4,

    ALL = std::numeric_limits<unsigned>::max()
  };

  inline Tag operator|(Tag a, Tag b)
  {
    return static_cast<Tag>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
  }

  inline Tag operator&(Tag a, Tag b)
  {
    return static_cast<Tag>(static_cast<unsigned>(a) & static_cast<unsigned>(b));
  }
}
