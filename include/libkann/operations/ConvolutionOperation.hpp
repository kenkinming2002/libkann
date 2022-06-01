#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ConvolutionOperation : public OperationImpl<ConvolutionOperation, 2>
  {
  public:
    ConvolutionOperation(size_t inputWidth, size_t inputHeight, size_t kernelSize);

  public:
    Tensor process_impl(inputs_t inputs) const;
    variables_t gradients_impl(variable_t gradient, variables_t inputs) const;

  private:
    size_t m_inputWidth, m_inputHeight;
    size_t m_kernelSize;
  };
}

