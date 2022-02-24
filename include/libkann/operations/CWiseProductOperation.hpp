#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class CWiseProductOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<VRef, VRef> gradientsImpl(VRef gradient, VRef input1, VRef input2) const override;
  };
}

