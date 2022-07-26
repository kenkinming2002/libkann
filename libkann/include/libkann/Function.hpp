#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace kann
{
  struct LIBKANN_EXPORT Function
  {
  public:
    virtual ~Function() = default;

  public:
    virtual tensor::Tensor<float> forward(tensor::Tensor<float> inputs) = 0;
    virtual tensor::Tensor<float> backward(tensor::Tensor<float> output_gradients) = 0;

  public:
    std::vector<tensor::Tensor<float>> saved_tensors;
  };
}
