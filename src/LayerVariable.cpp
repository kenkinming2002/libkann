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

  std::shared_ptr<const Variable> LayerVariable::insert(Type type, NewParameter parameter)
  {
    auto& map = this->map(type);
    auto [it, success] = map.emplace(std::move(parameter), std::make_shared<const Variable>());
    assert(success);
    return it->second;
  }

  std::shared_ptr<const Variable> LayerVariable::lookup(Type type, NewParameter parameter) const
  {
    auto& map = this->map(type);
    return map.at(parameter);
  }
}
