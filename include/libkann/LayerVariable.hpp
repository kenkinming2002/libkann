#pragma once

#include <libkann/Variable.hpp>
#include <libkann/QualifiedName.hpp>

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

    std::shared_ptr<const Variable> insert(Type type, QualifiedName qualifiedName);
    std::shared_ptr<const Variable> lookup(Type type, QualifiedName qualifiedName) const;
    void assign(Type type, QualifiedName qualifiedName, std::shared_ptr<const Variable> variable);

  public:
    std::shared_ptr<const Variable> variable;

    std::unordered_map<QualifiedName, std::shared_ptr<const Variable>> parameterVariables;
    std::unordered_map<QualifiedName, std::shared_ptr<const Variable>> stateVariables;
  };

}
