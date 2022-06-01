#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ReduceOperation : public Operation
  {
  public:
    ReduceOperation(size_t input_count);

  public:
    tensor_t process(std::vector<const Tensor*> inputs) const override;
    std::vector<variable_t> gradients(variable_t gradient, std::vector<variable_t> inputs) const override;

  private:
    size_t m_input_count;
  };
}
