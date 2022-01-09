#include <libkann/Variable.hpp>

#include <libkann/layers/IdentityLayer.hpp>

#include <memory>
#include <variant>

namespace kann
{
  std::shared_ptr<Variable> Variable::constant(size_t size)
  {
    auto result = std::make_shared<Variable>();
    result->size  = size;
    return result;
  }

  std::shared_ptr<Variable> operator|(std::shared_ptr<Variable> variable, std::shared_ptr<Layer> layer)
  {
    assert(variable->size == layer->inputSize());
    auto result = Variable::constant(layer->outputSize());
    result->inputs.emplace_back(std::move(variable), std::move(layer));
    return result;
  }

  std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs)
  {
    assert(lhs->size == rhs->size);
    auto result = Variable::constant(lhs->size);
    result->inputs.emplace_back(std::move(lhs), std::make_shared<IdentityLayer>(lhs->size, lhs->size, 0));
    result->inputs.emplace_back(std::move(rhs), std::make_shared<IdentityLayer>(rhs->size, rhs->size, 0));
    return result;
  }
}
