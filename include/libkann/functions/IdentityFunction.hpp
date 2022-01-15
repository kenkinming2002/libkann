#pragma once

#include <libkann/Function.hpp>

namespace kann
{
  class IdentityFunction : public UnaryFunction
  {
  protected:
    std::shared_ptr<const Variable> impl(std::shared_ptr<const Variable>) const override;
  };
}

