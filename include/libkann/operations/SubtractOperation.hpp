#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class SubtractOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<VRef, VRef> gradientsImpl(VRef gradient, VRef a, VRef b) const override;
  };
}


