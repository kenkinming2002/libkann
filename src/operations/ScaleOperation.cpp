#include <libkann/operations/ScaleOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  ScaleOperation::ScaleOperation(double val) : m_val(val) {}

  Tensor ScaleOperation::process_impl(inputs_t inputs) const
  {
    const auto& [input] = inputs;

    Tensor result(input->size());
    result.asArray() = input->asArray() * m_val;
    return result;
  }

  auto ScaleOperation::gradients_impl(variable_t gradient, variables_t) const -> variables_t
  {
    assert(m_val != 0.0);
    return {std::make_shared<const Variable>(std::vector{gradient}, std::make_shared<ScaleOperation>(m_val))};
  }
}
