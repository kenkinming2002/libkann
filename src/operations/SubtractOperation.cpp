#include <libkann/operations/SubtractOperation.hpp>

#include <libkann/operations/ScaleOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  Tensor SubtractOperation::process_impl(inputs_t inputs) const
  {
    const auto& [a, b] = inputs;
    assert(a->size() == b->size());
    Tensor result(a->size());
    result.asArray() = a->asArray() - b->asArray();
    return result;
  }

  auto SubtractOperation::gradients_impl(variable_t gradient, variables_t) const -> variables_t
  {
    return {gradient, std::make_shared<const Variable>(std::vector{gradient}, std::make_shared<ScaleOperation>(-1.0))};
  }
}

