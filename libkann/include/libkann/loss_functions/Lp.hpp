#pragma once

#include <libkann/Export.hpp>
#include <libkann/LossFunction.hpp>

namespace kann
{
  struct KANN_EXPORT LpLossFunction : public LossFunction
  {
  public:
    KANN_EXPORT LpLossFunction(unsigned p);

  public:
    KANN_EXPORT tensor::Tensor<float> forward(tensor::Tensor<float> inputs) override;
    KANN_EXPORT tensor::Tensor<float> backward(tensor::Tensor<float> output_gradients) override;

  private:
    unsigned m_p;
  };
}
