#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class TensorProductOperation : public Operation
  {
  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    operation_t differentiate() const override;
  };
}
