#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class IdentityOperation : public UnaryOperation
  {
  public:
    Tensor processImpl(const Tensor& input) const override;
    std::shared_ptr<const Function> function() const override;
  };
}

