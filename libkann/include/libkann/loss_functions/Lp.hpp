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
    KANN_EXPORT Tensor<const float> forward(Tensor<const float> inputs) override;
    KANN_EXPORT Tensor<const float> backward(Tensor<const float> output_gradients) override;

  private:
    unsigned m_p;
  };
}
