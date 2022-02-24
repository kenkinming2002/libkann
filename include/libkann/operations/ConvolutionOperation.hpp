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
    std::pair<VRef, VRef> gradientsImpl(VRef gradient, VRef input, VRef kernel) const override;

  private:
    size_t m_inputWidth, m_inputHeight;
    size_t m_kernelSize;
  };
}

