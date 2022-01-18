#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ReduceOperation : public Operation
  {
  public:
    ReduceOperation(size_t inputCount);

  public:
    std::shared_ptr<const Tensor> process(std::vector<std::shared_ptr<const Tensor>> inputs) const override;
    VariableList gradients(VariableHandle gradient, VariableList inputs) const override;

  private:
    size_t m_inputCount;
  };
}
