#include <libkann/operations/SubtractOperation.hpp>

#include <libkann/operations/MultiplyOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  Tensor SubtractOperation::processImpl(const Tensor& a, const Tensor& b) const
  {
    assert(a.size() == b.size());
    Tensor result(a.size());
    result.asArray() = a.asArray() - b.asArray();
    return result;
  }

  VariablePair SubtractOperation::gradientsImpl(VariableHandle gradient, VariableHandle a, VariableHandle b) const
  {
    return std::make_pair(gradient, std::make_shared<const Variable>(std::vector{gradient}, std::make_shared<MultiplyOperation>(-1.0)));
  }
}

