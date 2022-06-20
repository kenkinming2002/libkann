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
    KANN_EXPORT Tensor forward(Tensor inputs) override;
    KANN_EXPORT Tensor backward(Tensor output_gradients) override;

  private:
    unsigned m_p;
  };
}
