#pragma once

#include <libkann/Export.hpp>

#include <libkann/Operation.hpp>

namespace kann
{
  class KANN_EXPORT TensorProductOperation : public Operation
  {
  public:
    KANN_EXPORT TensorProductOperation(size_t rank_m, size_t rank_n, size_t rank_k);

  public:
    KANN_EXPORT std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    KANN_EXPORT operation_t differentiate() const override;

  private:
    size_t m_rank_m, m_rank_n, m_rank_k;
  };
}
