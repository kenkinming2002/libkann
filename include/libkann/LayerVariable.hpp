#pragma once

#include <libkann/Variable.hpp>
#include <libkann/NewParameter.hpp>

#include <unordered_map>
#include <memory>

namespace kann
{
  struct LayerVariable
  {
  public:
    enum class Type { PARAMETER, STATE };

    auto& map(Type type);
    const auto& map(Type type) const;

    std::shared_ptr<const Variable> insert(Type type, NewParameter parameter);
    std::shared_ptr<const Variable> lookup(Type type, NewParameter parameter) const;

  public:
    std::shared_ptr<const Variable> variable;

    std::unordered_map<NewParameter, std::shared_ptr<const Variable>> parameterVariables;
    std::unordered_map<NewParameter, std::shared_ptr<const Variable>> stateVariables;
  };

}
