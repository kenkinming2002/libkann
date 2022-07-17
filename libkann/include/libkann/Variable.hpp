#pragma once

#include <libtensor/Tensor.hpp>
#include <libtensor/Initializer.hpp>

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
    static Variable create_constant(tensor::Shape shape, float value)
    {
      return Variable{
        .shape = shape,
        .value = tensor::create_constant(shape, value)
      };
    }

    static Variable create_normal(tensor::Shape shape, float mean, float stddev, std::default_random_engine& prng)
    {
      return Variable{
        .shape = shape,
        .value = tensor::create_normal(shape, mean, stddev, prng)
      };
    }

    static Variable create_uniform(tensor::Shape shape, float a, float b, std::default_random_engine& prng)
    {
      return Variable{
        .shape = shape,
        .value = tensor::create_uniform(shape, a, b, prng)
      };
    }
  };
}
