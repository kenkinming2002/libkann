#pragma once

#include <libkann/Export.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    LIBKANN_EXPORT virtual ~Optimizer() = default;

  public:
    LIBKANN_EXPORT virtual void step() {}
    LIBKANN_EXPORT virtual void optimize(Variable& variable) const = 0;
  };
}
