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
    virtual tensor::Tensor<const float> forward(tensor::Tensor<const float> inputs) = 0;
    virtual tensor::Tensor<const float> backward(tensor::Tensor<const float> output_gradients) = 0;

  public:
    std::vector<tensor::Tensor<const float>> saved_tensors;
  };
}
