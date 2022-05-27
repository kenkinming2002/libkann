#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ConvolutionOperation : public BinaryOperation
  {
  public:
    ConvolutionOperation(size_t inputWidth, size_t inputHeight, size_t kernelSize);

  public:
    Tensor processImpl(const Tensor& input, const Tensor& kernel) const override;
    std::pair<variable_t, variable_t> gradientsImpl(variable_t gradient, variable_t input, variable_t kernel) const override;

  private:
    size_t m_inputWidth, m_inputHeight;
    size_t m_kernelSize;
  };
}

