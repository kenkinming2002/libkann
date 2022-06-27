#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace kann
{
  struct Function
  {
  public:
    virtual ~Function() = default;

  public:
    virtual Tensor<const float> forward(Tensor<const float> inputs) = 0;
    virtual Tensor<const float> backward(Tensor<const float> output_gradients) = 0;

  public:
    std::vector<Tensor<const float>> saved_tensors;
  };
}
