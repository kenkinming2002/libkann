#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class DeconvolutionOperation : public BinaryOperation
  {
  public:
    DeconvolutionOperation(size_t inputWidth, size_t inputHeight, size_t kernelSize);

  public:
    Tensor processImpl(const Tensor& input, const Tensor& kernel) const override;
    std::pair<CRef<Variable>, CRef<Variable>> gradientsImpl(CRef<Variable> gradient, CRef<Variable> input, CRef<Variable> kernel) const override;

  private:
    size_t m_inputWidth, m_inputHeight;
    size_t m_kernelSize;
  };
}


