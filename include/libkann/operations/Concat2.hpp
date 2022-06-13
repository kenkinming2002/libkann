#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class Concat2Operation : public Operation
  {
  public:
    Concat2Operation(size_t size1, size_t size2);

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override;
    operation_t differentiate() const override;

  private:
    size_t m_size1, m_size2;
  };
}


