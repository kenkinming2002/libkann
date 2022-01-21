#pragma once

#include <libkann/operations/CWiseProductOperation.hpp>
#include <libkann/Operation.hpp>

namespace kann
{
  struct AssertFunc
  {
    double operator()(double val) const { assert(false); }
  };

  template<typename ForwardFunc, typename BackwardFunc>
  class CWiseOperation : public UnaryOperation
  {
  public:
    CWiseOperation(ForwardFunc forwardFunc = ForwardFunc(), BackwardFunc backwardFunc = BackwardFunc())
      : m_forwardFunc(forwardFunc), m_backwardFunc(backwardFunc) {}

  public:
    Tensor processImpl(const Tensor& input) const override
    {
      Tensor result(input.size());
      result.asArray() = input.asArray().unaryExpr(m_forwardFunc);
      return result;
    }

    VariableHandle gradientsImpl(VariableHandle gradient, VariableHandle input) const override
    {
      const auto tmp = std::make_shared<const Variable>(std::vector{input}, std::make_shared<CWiseOperation<BackwardFunc, AssertFunc>>(m_backwardFunc, AssertFunc()));
      const auto result = std::make_shared<const Variable>(std::vector{gradient, tmp}, std::make_shared<CWiseProductOperation>());
      return result;
    }

  private:
    [[no_unique_address]] ForwardFunc m_forwardFunc;
    [[no_unique_address]] BackwardFunc m_backwardFunc;
  };
}
