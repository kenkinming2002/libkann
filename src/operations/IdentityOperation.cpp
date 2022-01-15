#include <libkann/operations/IdentityOperation.hpp>

#include <libkann/functions/IdentityFunction.hpp>


namespace kann
{
  Tensor IdentityOperation::processImpl(const Tensor& input) const
  {
    return input;
  }

  std::shared_ptr<const Function> IdentityOperation::function() const
  {
    static const auto identityFunction = std::make_shared<IdentityFunction>();
    return identityFunction;
  }
}
