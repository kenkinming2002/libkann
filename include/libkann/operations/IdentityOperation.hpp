#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class IdentityOperation : public OperationImpl<IdentityOperation, 1>
  {
  public:
    IdentityOperation(size_t inputSize, size_t outputSize, size_t offset);

  public:
    Tensor process_impl(inputs_t inputs) const;
    variables_t gradients_impl(variable_t gradient, variables_t inputs) const;

  private:
    size_t m_inputSize;
    size_t m_outputSize;
    size_t m_offset;
  };
}

