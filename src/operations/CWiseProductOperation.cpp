#include <libkann/operations/CWiseProductOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  Tensor CWiseProductOperation::processImpl(const Tensor& a, const Tensor& b) const
  {
    assert(a.size() == b.size());
    Tensor result(a.size());
    result.asArray() = a.asArray().cwiseProduct(b.asArray());
    return result;
  }

  std::pair<CRef<Variable>, CRef<Variable>> CWiseProductOperation::gradientsImpl(CRef<Variable> gradient, CRef<Variable> input1, CRef<Variable> input2) const
  {
    return std::make_pair(
      std::make_shared<const Variable>(std::vector{gradient, input2}, std::make_shared<CWiseProductOperation>()),
      std::make_shared<const Variable>(std::vector{gradient, input2}, std::make_shared<CWiseProductOperation>())
    );
  }
}

