#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ReduceOperation : public Operation
  {
  public:
    ReduceOperation(size_t inputCount);

  public:
    Tensor process(const std::vector<Tensor>& inputs) const override;

  private:
    size_t m_inputCount;
  };
}
