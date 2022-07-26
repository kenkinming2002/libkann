#pragma once

#include <libkann/Export.hpp>
#include <libkann/LossFunction.hpp>

namespace kann
{
  struct LIBKANN_EXPORT LpLossFunction : public LossFunction
  {
  public:
    LIBKANN_EXPORT LpLossFunction(unsigned p);

  public:
    LIBKANN_EXPORT tensor::Tensor<float> forward(tensor::Tensor<float> inputs) override;
    LIBKANN_EXPORT tensor::Tensor<float> backward(tensor::Tensor<float> output_gradients) override;

  private:
    unsigned m_p;
  };
}
