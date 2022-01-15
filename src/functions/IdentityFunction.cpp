#include <libkann/functions/IdentityFunction.hpp>

#include <libkann/operations/IdentityOperation.hpp>

namespace kann
{
  std::shared_ptr<const Variable> IdentityFunction::impl(std::shared_ptr<const Variable> v) const
  {
    static const auto identityOperation = std::make_shared<IdentityOperation>();
    return std::make_shared<const Variable>(std::vector{v}, identityOperation);
  }
}
