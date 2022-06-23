#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tensor.hpp>

namespace kann
{
  struct Function
  {
  public:
    virtual ~Function() = default;

  public:
    virtual Tensor<float> forward(Tensor<float> inputs) = 0;
    virtual Tensor<float> backward(Tensor<float> output_gradients) = 0;

  public:
    std::vector<Tensor<float>> saved_tensors;
  };
}
