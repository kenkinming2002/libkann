#pragma once

#include <libkann/Types.hpp>

#include <vector>

namespace kann
{
  /* A variable could be an adjustable constant or result of applying operation
   * on a list of other variables */
  class Variable
  {
  public:
    template<typename T>
    static variable_t apply(T operation, std::vector<variable_t> inputs)
    {
      return std::make_shared<const Variable>(
          std::move(inputs),
          std::make_shared<T>(std::move(operation))
      );
    }

  public:
    Variable() = default;
    Variable(std::vector<variable_t> inputs, operation_t op)
      : inputs(std::move(inputs)), op(std::move(op)) {}

  public:
    std::vector<variable_t> inputs;
    operation_t op;
  };
}
