#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class SubtractOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<CRef<Variable>, CRef<Variable>> gradientsImpl(CRef<Variable> gradient, CRef<Variable> a, CRef<Variable> b) const override;
  };
}


