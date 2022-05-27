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

  std::pair<variable_t, variable_t> CWiseProductOperation::gradientsImpl(variable_t gradient, variable_t input1, variable_t input2) const
  {
    return std::make_pair(
      std::make_shared<const Variable>(std::vector{gradient, input2}, std::make_shared<CWiseProductOperation>()),
      std::make_shared<const Variable>(std::vector{gradient, input2}, std::make_shared<CWiseProductOperation>())
    );
  }
}

