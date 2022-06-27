#pragma once

#include <libtensor/Tensor.hpp>

#include <optional>

namespace kann
{
  struct Variable
  {
  public:
    tensor::Shape shape;

  public:
    tensor::Tensor<float> value;
    tensor::Tensor<float> gradient;

  public:
    std::vector<tensor::Tensor<float>> optimizer_states;

  public:
    static Variable create(tensor::Shape shape)
    {
      return Variable{
        .shape = shape,
        .value    = tensor::Tensor<float>::create(shape),
        .gradient = tensor::Tensor<float>::create(shape),
        .optimizer_states = {}
      };
    }
  };
}
