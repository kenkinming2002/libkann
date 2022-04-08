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
    template<typename T>
    static std::shared_ptr<const Variable> apply(T operation, std::vector<std::shared_ptr<const Variable>> inputs)
    {
      return std::make_shared<const Variable>(
          std::move(inputs),
          std::make_shared<T>(std::move(operation))
      );
    }

  public:
    Variable() = default;
    Variable(std::vector<CRef<Variable>> inputs, CRef<Operation> op)
      : inputs(std::move(inputs)), op(std::move(op)) {}

  public:
    std::vector<CRef<Variable>> inputs;
    CRef<Operation> op;
  };
}
