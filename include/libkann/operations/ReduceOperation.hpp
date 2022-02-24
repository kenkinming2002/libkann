#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ReduceOperation : public Operation
  {
  public:
    ReduceOperation(size_t inputCount);

  public:
    std::shared_ptr<const Tensor> process(std::vector<const Tensor*> inputs) const override;
    std::vector<VRef> gradients(VRef gradient, std::vector<VRef> inputs) const override;

  private:
    size_t m_inputCount;
  };
}
