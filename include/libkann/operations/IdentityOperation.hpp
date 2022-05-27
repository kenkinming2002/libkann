#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class IdentityOperation : public UnaryOperation
  {
  public:
    IdentityOperation(size_t inputSize, size_t outputSize, size_t offset);

  public:
    Tensor processImpl(const Tensor& input) const override;
    variable_t gradientsImpl(variable_t gradient, variable_t) const override;

  private:
    size_t m_inputSize;
    size_t m_outputSize;
    size_t m_offset;
  };
}

