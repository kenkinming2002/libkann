#include <libkann/LayerVariable.hpp>

#include <libkann/Variable.hpp>

#include <assert.h>

namespace kann
{
  auto& LayerVariable::map(Type type)
  {
    switch(type)
    {
    case Type::PARAMETER:
      return parameterVariables;
    case Type::STATE:
      return stateVariables;
    default:
      assert(false && "Unreachable");
    }
  }

  const auto& LayerVariable::map(Type type) const
  {
    switch(type)
    {
    case Type::PARAMETER:
      return parameterVariables;
    case Type::STATE:
      return stateVariables;
    default:
      assert(false && "Unreachable");
    }
  }

  std::shared_ptr<const Variable> LayerVariable::insert(Type type, QualifiedName qualifiedName)
  {
    auto& map = this->map(type);
    auto [it, success] = map.emplace(std::move(qualifiedName), std::make_shared<const Variable>());
    assert(success);
    return it->second;
  }

  std::shared_ptr<const Variable> LayerVariable::lookup(Type type, QualifiedName qualifiedName) const
  {
    auto& map = this->map(type);
    return map.at(qualifiedName);
  }

  void LayerVariable::assign(Type type, QualifiedName qualifiedName, std::shared_ptr<const Variable> variable)
  {
    auto& map = this->map(type);
    map.at(qualifiedName) = std::move(variable);
  }
}
