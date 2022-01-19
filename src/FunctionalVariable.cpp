#include <libkann/FunctionalVariable.hpp>

#include <libkann/layers/IdentityLayer.hpp>

#include <memory>
#include <variant>

namespace kann
{
  std::shared_ptr<FunctionalVariable> FunctionalVariable::constant(size_t size)
  {
    auto result = std::make_shared<FunctionalVariable>();
    result->size  = size;
    return result;
  }

  std::shared_ptr<FunctionalVariable> operator|(std::shared_ptr<FunctionalVariable> variable, std::shared_ptr<Layer> layer)
  {
    assert(variable->size == layer->inputSize());
    auto result = FunctionalVariable::constant(layer->outputSize());
    result->inputs.emplace_back(std::move(variable), std::move(layer));
    return result;
  }

  std::shared_ptr<FunctionalVariable> operator+(std::shared_ptr<FunctionalVariable> lhs, std::shared_ptr<FunctionalVariable> rhs)
  {
    assert(lhs->size == rhs->size);
    auto result = FunctionalVariable::constant(lhs->size);
    result->inputs.emplace_back(std::move(lhs), std::make_shared<IdentityLayer>(lhs->size, lhs->size, 0));
    result->inputs.emplace_back(std::move(rhs), std::make_shared<IdentityLayer>(rhs->size, rhs->size, 0));
    return result;
  }
}
