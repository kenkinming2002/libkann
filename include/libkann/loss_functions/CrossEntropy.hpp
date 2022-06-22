#pragma once

#include <libkann/Export.hpp>
#include <libkann/LossFunction.hpp>

namespace kann
{
  struct KANN_EXPORT CrossEntropyLossFunction : public LossFunction
  {
  public:
    KANN_EXPORT Tensor forward(Tensor inputs) override;
    KANN_EXPORT Tensor backward(Tensor output_gradients) override;
  };
}

