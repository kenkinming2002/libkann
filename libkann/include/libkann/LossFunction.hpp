#pragma once

#include <libkann/Export.hpp>
#include <libkann/Function.hpp>

#include <optional>

namespace kann
{
  struct KANN_EXPORT LossFunction : public Function
  {
  public:
    std::optional<tensor::Tensor<const float>> expected_outputs;

  public:
    KANN_EXPORT tensor::Tensor<const float> forward(tensor::Tensor<const float> inputs) override;
    KANN_EXPORT tensor::Tensor<const float> backward(tensor::Tensor<const float> output_gradients) override;

  public:
    // Note: inputs are 2D tensor while output_gradients is 1D
    KANN_EXPORT virtual tensor::Tensor<const float> loss_forward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs) { assert(false && "Unimplemented"); };
    KANN_EXPORT virtual tensor::Tensor<const float> loss_backward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs, tensor::Tensor<const float> output_gradients) { assert(false && "Unimplemented"); };
  };
}
