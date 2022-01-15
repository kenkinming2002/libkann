#pragma once

#include <libkann/Function.hpp>

namespace kann
{
  class MatrixMultiplyFunction : public BinaryFunction
  {
  public:
    /* Mutiply a mxk matrix with a kxn matrix.
     *
     * @param transpose 1 whether a first matrix(matrix with dimension mxk)
     *        should be transposed
     * @param transpose 1 whether a second matrix(matrix with dimension kxn)
     *        should be transposed */
    MatrixMultiplyFunction(size_t m, size_t n, size_t k, bool transpose1, bool transpose2);

  protected:
    std::shared_ptr<const Variable> impl(std::shared_ptr<const Variable>, std::shared_ptr<const Variable>) const override;

  private:
    size_t m_m, m_n, m_k;
    bool m_transpose1, m_transpose2;
  };
}


