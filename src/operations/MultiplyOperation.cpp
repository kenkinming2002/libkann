#include <libkann/operations/MultiplyOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  MultiplyOperation::MultiplyOperation(double val) : m_val(val) {}

  Tensor MultiplyOperation::processImpl(const Tensor& input) const
  {
    Tensor result(input.size());
    result.asArray() = input.asArray() * m_val;
    return result;
  }

  VRef MultiplyOperation::gradientsImpl(VRef gradient, VRef input) const
  {
    assert(m_val != 0.0);
    return std::make_shared<const Variable>(std::vector{gradient}, std::make_shared<MultiplyOperation>(1.0/m_val));
  }
}
