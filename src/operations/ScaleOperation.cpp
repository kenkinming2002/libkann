#include <libkann/operations/ScaleOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  ScaleOperation::ScaleOperation(double val) : m_val(val) {}

  Tensor ScaleOperation::processImpl(const Tensor& input) const
  {
    Tensor result(input.size());
    result.asArray() = input.asArray() * m_val;
    return result;
  }

  variable_t ScaleOperation::gradientsImpl(variable_t gradient, variable_t input) const
  {
    assert(m_val != 0.0);
    return std::make_shared<const Variable>(std::vector{gradient}, std::make_shared<ScaleOperation>(m_val));
  }
}
