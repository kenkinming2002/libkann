#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ReduceOperation : public Operation
  {
  public:
    ReduceOperation(size_t inputCount);

  public:
    Tensor process(std::vector<std::reference_wrapper<const Tensor>> inputs) const override;
    VariableList gradients(VariableHandle gradient, VariableList inputs) const override;

  private:
    size_t m_inputCount;
  };
}
