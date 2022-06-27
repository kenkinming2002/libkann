#pragma once

#include <libtensor/Tensor.hpp>

#include <optional>

namespace kann
{
  struct Variable
  {
  public:
    Shape shape;

  public:
    Tensor<float> value;
    Tensor<float> gradient;

  public:
    std::vector<Tensor<float>> optimizer_states;

  public:
    static Variable create(Shape shape)
    {
      return Variable{
        .shape = shape,
        .value    = Tensor<float>::create(shape),
        .gradient = Tensor<float>::create(shape),
        .optimizer_states = {}
      };
    }
  };
}
