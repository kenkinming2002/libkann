#include <libkann/operations/IdentityOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  Tensor IdentityOperation::processImpl(const Tensor& input) const
  {
    return input;
  }

  VariableHandle IdentityOperation::gradientsImpl(VariableHandle gradient, VariableHandle) const
  {
    return gradient;
  }
}
