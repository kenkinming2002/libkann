#pragma once

#include <libkann/Tensor.hpp>
#include <libkann/Operation.hpp>

#include <optional>

namespace kann
{
  /* A variable could be an adjustable constant or result of applying operation
   * on a list of other variables */
  class Variable
  {
  public:
    Variable() = default;
    Variable(std::vector<std::shared_ptr<const Variable>> inputs, std::shared_ptr<const Operation> op)
      : inputs(std::move(inputs)), op(std::move(op)) {}

  public:
    std::vector<std::shared_ptr<const Variable>> inputs;
    std::shared_ptr<const Operation> op;
  };
}
