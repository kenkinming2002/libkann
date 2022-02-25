#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ReduceOperation : public Operation
  {
  public:
    ReduceOperation(size_t inputCount);

  public:
    CRef<Tensor> process(std::vector<const Tensor*> inputs) const override;
    std::vector<CRef<Variable>> gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs) const override;

  private:
    size_t m_inputCount;
  };
}
