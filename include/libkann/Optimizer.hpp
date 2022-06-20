#pragma once

#include <libkann/Export.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    KANN_EXPORT virtual ~Optimizer() = default;

  public:
    KANN_EXPORT virtual void step() {}
    KANN_EXPORT virtual void optimize(Variable& variable) const = 0;
  };
}
