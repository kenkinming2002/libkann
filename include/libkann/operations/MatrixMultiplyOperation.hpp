#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class MatrixMultiplyOperation : public BinaryOperation
  {
  public:
    /* Mutiply a mxk matrix with a kxn matrix.
     *
     * @param transpose 1 whether a first matrix(matrix with dimension mxk)
     *        should be transposed
     * @param transpose 1 whether a second matrix(matrix with dimension kxn)
     *        should be transposed */
    MatrixMultiplyOperation(size_t m, size_t n, size_t k, bool transpose1, bool transpose2);

  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<variable_t, variable_t> gradientsImpl(variable_t gradient, variable_t a, variable_t b) const override;

  private:
    size_t m_m, m_n, m_k;
    bool m_transpose1, m_transpose2;
  };
}
