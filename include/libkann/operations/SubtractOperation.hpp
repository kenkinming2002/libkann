#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class SubtractOperation : public Operation
  {
  public:
    SubtractOperation(size_t size);

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override;
    operation_t differentiate() const override;

  private:
    size_t m_size;
  };
}


