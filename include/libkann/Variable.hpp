#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <optional>

namespace kann
{
  class Operation;

  /* A variable could be an adjustable constant or result of applying operation
   * on a list of other variables */
  class Variable
  {
  public:
    Variable() = default;
    Variable(std::vector<CRef<Variable>> inputs, CRef<Operation> op)
      : inputs(std::move(inputs)), op(std::move(op)) {}

  public:
    std::vector<CRef<Variable>> inputs;
    CRef<Operation> op;
  };
}
