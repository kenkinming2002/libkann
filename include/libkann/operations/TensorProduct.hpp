#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class TensorProductOperation : public Operation
  {
  public:
    TensorProductOperation(size_t rank_m, size_t rank_n, size_t rank_k);

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    operation_t differentiate() const override;

  private:
    size_t m_rank_m, m_rank_n, m_rank_k;
  };
}
