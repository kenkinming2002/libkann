#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class AddOperation : public Operation
  {
  public:
    AddOperation(Shape shape);

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    operation_t differentiate() const override;

  private:
    Shape m_shape;
  };
}



