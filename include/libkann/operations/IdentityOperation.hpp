#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class IdentityOperation : public UnaryOperation
  {
  public:
    Tensor processImpl(const Tensor& input) const override;
    VariableHandle gradientsImpl(VariableHandle gradient, VariableHandle) const override;
  };
}

