#pragma once

#include <libkann/Export.hpp>
#include <libkann/LossFunction.hpp>

namespace kann
{
  struct KANN_EXPORT CrossEntropyLossFunction : public LossFunction
  {
  public:
    KANN_EXPORT tensor::Tensor<const float> loss_forward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs) override;
    KANN_EXPORT tensor::Tensor<const float> loss_backward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs, tensor::Tensor<const float> output_gradients) override;
  };
}

